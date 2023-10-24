import pickle

import numpy as np
import multiprocessing as mp
import torch
from torch.utils.data import Dataset
from dgl import transforms as T
from utils import (
    build_knns,
    build_next_level,
    decode,
    density_estimation,
    fast_knns2spmat,
    knns2ordered_nbrs,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
    mark_same_camera_nbrs,
)

import dgl

def worker(features, labels, xws, yws, cam_ids, k, levels, device, faiss_gpu):
    dataset = LanderDataset(
        features=features,
        labels=labels,
        xws=xws,
        yws=yws,
        cam_ids=cam_ids,
        k=k,
        levels=levels,
        faiss_gpu=faiss_gpu,
        augment=True
    )
    return [g.to(device) for g in dataset.gs]

def prepare_dataset_graphs_mp(data_path, k, levels, device, faiss_gpu, num_workers):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    g_labels = []
    g_features = []
    g_xws = []
    g_yws = []
    g_cams = []
    process_inputs = []
    for scene in data:
        # Embed world position into embeddings
        node_len = scene['node_embeds'].shape[-1]
        coo_extend = node_len // 2
        xws = np.repeat(scene['xws'][:, None] / 360., coo_extend, axis=-1)
        yws = np.repeat(scene['yws'][:, None] / 288., coo_extend, axis=-1)
        scene['node_embeds'] = np.concatenate([scene['node_embeds'], xws, yws], axis=-1)

        # scene['node_embeds'] = np.concatenate(
        #     [scene['node_embeds'], scene['xws'][:, None] / 360., scene['yws'][:, None] / 288.],
        #     axis=-1
        # )

        scene['node_embeds'] = l2norm(scene['node_embeds'].astype("float32"))
        g_labels.append(scene['node_labels'])
        g_features.append(scene['node_embeds'])
        g_xws.append(scene['xws'])
        g_yws.append(scene['yws'])
        g_cams.append(scene['cam_ids'])
        process_inputs.append((
            scene['node_embeds'],
            scene['node_labels'],
            scene['xws'],
            scene['yws'],
            scene['cam_ids'],
            k,
            levels,
            device,
            faiss_gpu
        ))

    with mp.Pool(num_workers) as pool:
        gss = pool.starmap(worker, process_inputs)

    return GraphDataset(gss, g_labels, g_features, g_xws, g_yws, g_cams)

class LanderDataset(object):
    def __init__(
        self,
        features,# (669560, 128)
        labels,# (669560,)
        xws,
        yws,
        cam_ids,
        cluster_features=None,
        k=10,
        levels=1,
        faiss_gpu=False,
        augment=False
    ):
        self.k = k
        self.augment = augment
        self.gs = []
        self.nbrs = []
        self.sims = []
        self.levels = levels
        global_xws = xws.copy()
        global_yws = yws.copy()

        # Augment graph transformations
        self.transforms = T.Compose([T.DropNode(p=0.05),
                                     T.DropEdge(p=0.05)])

        # Initialize features and labels
        features = l2norm(features.astype("float32"))
        if features.shape[0] <= self.k:
            self.k = max(features.shape[0] - 1, 2)

        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.int_)
        ids = np.arange(global_num_nodes)

        # Recursive graph construction
        # print(features.shape)
        for lvl in range(self.levels):
            if features.shape[0] < self.k:
                self.levels = lvl
                break

            knns = build_knns(features, self.k, faiss_gpu)
            knns = mark_same_camera_nbrs(knns, cam_ids)

            nbrs, sims = knns[:, 0, :].astype(np.int32), knns[:, 1, :]

            self.nbrs.append(nbrs)
            self.sims.append(sims)
            density = density_estimation(sims, nbrs, labels)

            g = self._build_graph(
                features, cluster_features, labels, xws, yws, cam_ids, density, knns
            )

            # Apply graph augmentation during training
            if self.augment:
                g = self.transforms(g)
                if g.num_nodes() == 0:
                    break

            self.gs.append(g)

            if lvl >= self.levels - 1:
                break

            # Decode peak nodes
            (
                new_pred_labels,
                peaks,
                global_edges,
                global_pred_labels,
                global_peaks,
            ) = decode(
                g,
                -np.inf,
                "sim",
                True,
                ids,
                global_edges,
                global_num_nodes,
                global_peaks,
            )
            ids = ids[peaks]
            features, labels, cam_ids, cluster_features, xws, yws = build_next_level(
                features,
                labels,
                peaks,
                cam_ids,
                global_features,
                global_pred_labels,
                global_peaks,
                global_xws,
                global_yws,
            )

            # If all peaks have same camera id, break
            if len(np.unique(cam_ids)) == 1:
                break

    def _build_graph(self, features, cluster_features, labels, xws, yws, cam_ids, density, knns):
        adj = fast_knns2spmat(knns, self.k)# adj sparse matrix (669560, 669560)
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)
        g = dgl.graph((indices[1], indices[0]), num_nodes=len(knns))
        g.ndata["features"] = torch.FloatTensor(features)
        g.ndata["cluster_features"] = torch.FloatTensor(cluster_features)
        g.ndata["labels"] = torch.LongTensor(labels)
        g.ndata["density"] = torch.FloatTensor(density)
        g.ndata["xws"] = torch.FloatTensor(xws)
        g.ndata["yws"] = torch.FloatTensor(yws)
        g.ndata["cam_ids"] = torch.LongTensor(cam_ids)
        g.edata["affine"] = torch.FloatTensor(values)
        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata["global_eid"] = g.edges(form="eid")
        g.ndata["norm"] = torch.FloatTensor(adj_row_sum)
        g.apply_edges(
            lambda edges: {
                "raw_affine": edges.data["affine"] / edges.dst["norm"]
            }
        )
        g.apply_edges(
            lambda edges: {
                "labels_conn": (
                    edges.src["labels"] == edges.dst["labels"]
                ).long()
            }
        )
        g.apply_edges(
            lambda edges: {
                "mask_conn": (
                    edges.src["density"] <= edges.dst["density"]
                ).bool()
            }
        )

        return g

class GraphDataset(Dataset):
    def __init__(self, gs, g_labels, g_features, g_xws, g_yws, g_cams):
        self.gs = gs
        self.g_labels = g_labels
        self.g_features = g_features
        self.g_xws = g_xws
        self.g_yws = g_yws
        self.g_cams = g_cams
    
    def __getitem__(self, index):
        sample = {
            'graphs': self.gs[index],
            'labels': self.g_labels[index],
            'features': self.g_features[index],
            'xws': self.g_xws[index],
            'yws': self.g_yws[index],
            'cam_ids': self.g_cams[index]
        }
        return sample
    
    def __len__(self):
        return len(self.gs)