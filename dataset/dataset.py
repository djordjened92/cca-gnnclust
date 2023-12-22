import numpy as np
import torch
import multiprocessing as mp
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset

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
        faiss_gpu=faiss_gpu
    )
    return [g.to(device) for g in dataset.gs]

def prepare_dataset_graphs_mp(sequence,
                              k,
                              levels,
                              faiss_gpu,
                              device,
                              num_workers):
    process_inputs = []
    for scene in sequence:
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

    return gss

class GraphDataset(Dataset):
    def __init__(self, scene_seq, gss):
        self.scene_seq = scene_seq
        self.gss = gss

    def __len__(self):
        return len(self.scene_seq)
    
    def __getitem__(self, index):
        scene = self.scene_seq[index]
        scene['graphs'] = self.gss[index]
        return scene

class SceneDataset(Dataset):
    def __init__(
            self,
            sequence,
            coo2meter,
            feature_model,
            device,
            transform=None
    ):
        self.coo2meter = coo2meter
        self.sequence = sequence
        self.transform = transform
        self.feature_model = feature_model
        self.device = device
    
    def __getitem__(self, index):
        scene = self.sequence[index]
        crop_paths = scene['bboxes_paths']
        embeds = []

        img_batch = []
        for path in crop_paths:
            crop_image = io.imread(path)
            crop_image = Image.fromarray(crop_image)
            if self.transform:
                crop_image = self.transform(crop_image)

            img_batch.append(crop_image)
        img_batch = torch.stack(img_batch, dim=0)

        # Extract features
        with torch.no_grad():
            _, embeds = self.feature_model(img_batch.to(self.device))
            embeds = embeds.cpu().numpy()

        # Embed box world coordinates
        xws = scene['xws'][:, None]# / self.coo2meter
        yws = scene['yws'][:, None]# / self.coo2meter

        node_embeds = embeds
        sample = {
            'node_labels': scene['node_labels'],
            'node_embeds': node_embeds,
            'xws': xws,
            'yws': yws,
            'cam_ids': scene['cam_ids']
        }
        return sample
    
    def __len__(self):
        return len(self.sequence)

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
        faiss_gpu=False
    ):
        self.k = k
        self.gs = []
        self.nbrs = []
        self.sims = []
        self.levels = levels
        global_xws = xws.copy()
        global_yws = yws.copy()

        # Initialize features and labels
        features = l2norm(features.astype("float32"))
        if features.shape[0] <= self.k:
            self.k = max(features.shape[0], 2)

        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.int_)
        ids = np.arange(global_num_nodes)

        # Recursive graph construction
        for lvl in range(self.levels):
            if features.shape[0] < self.k:
                self.levels = lvl
                break

            knns = build_knns(features, self.k, xws, yws, faiss_gpu)
            knns = mark_same_camera_nbrs(knns, cam_ids)

            nbrs, sims = knns[:, 0, :].astype(np.int32), knns[:, 1, :]

            self.nbrs.append(nbrs)
            self.sims.append(sims)
            density = density_estimation(sims, nbrs, labels)

            g = self._build_graph(
                features, cluster_features, labels, xws, yws, cam_ids, density, knns
            )

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
