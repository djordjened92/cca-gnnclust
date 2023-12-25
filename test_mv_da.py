import os
import argparse
import yaml
import pickle
import multiprocessing as mp

import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from models import LANDER, load_feature_extractor
from dataset import SceneDataset
from dataset import LanderDataset
from models import LANDER
from utils import build_next_level, decode, stop_iterating, l2norm, metrics

def inference(features, labels, xws, yws, coo2meter, cam_ids, model, device, args):
    # Initialize objects
    global_xws = xws.copy()
    global_yws = yws.copy()
    global_features = features.copy()
    dataset = LanderDataset(
        features=features,
        labels=labels,
        xws=global_xws,
        yws=global_yws,
        coo2meter=coo2meter,
        cam_ids=cam_ids,
        k=args.knn_k,
        levels=1,
        faiss_gpu=args.faiss_gpu,
    )
    g = dataset.gs[0].to(device)
    g.ndata["pred_den"] = torch.zeros((g.num_nodes()), device=device)
    g.edata["prob_conn"] = torch.zeros((g.num_edges(), 2), device=device)
    ids = np.arange(g.num_nodes())
    global_edges = ([], [])
    global_edges_len = len(global_edges[0])
    global_num_nodes = g.num_nodes()
    prev_global_pred_labels = []
    num_edges_add_last_level = np.Inf
    _, cam_counts = np.unique(cam_ids, return_counts=True)

    # Maximum instances per camera is number of clusters
    max_inst_count = np.max(cam_counts)

    # print(f'\nLables: {labels}')
    # Predict connectivity and density
    for level in range(args.levels):
        with torch.no_grad():
            g = model(g)
        (
            new_pred_labels,
            peaks,
            global_edges,
            global_pred_labels,
            global_peaks,
        ) = decode(
            g,
            args.tau,
            args.threshold,
            False,
            ids,
            global_edges,
            global_num_nodes,
        )
        ids = ids[peaks]
        new_global_edges_len = len(global_edges[0])
        num_edges_add_this_level = new_global_edges_len - global_edges_len
        # print(f'Level {level}, pred labels: {global_pred_labels}')

        if len(np.unique(global_pred_labels)) <= max_inst_count \
            and len(prev_global_pred_labels) > 0:
            return prev_global_pred_labels

        prev_global_pred_labels = global_pred_labels

        if stop_iterating(
            level,
            args.levels,
            args.early_stop,
            num_edges_add_this_level,
            num_edges_add_last_level,
            args.knn_k,
        ):
            break
        global_edges_len = new_global_edges_len
        num_edges_add_last_level = num_edges_add_this_level

        # build new dataset
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
        # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
        dataset = LanderDataset(
            features=features,
            labels=labels,
            xws=xws,
            yws=yws,
            coo2meter=coo2meter,
            cam_ids=cam_ids,
            k=args.knn_k,
            levels=1,
            faiss_gpu=False,
            cluster_features=cluster_features,
        )
        if len(dataset.gs) == 0:
            break
        g = dataset.gs[0].to(device)

    return global_pred_labels

def main(args, device):
    # Load config
    config = yaml.safe_load(open('config/config_training.yaml', 'r'))

    #################################
    # Load feature extraction model #
    #################################
    feature_model = load_feature_extractor(config, device)

    #############
    # Load data #
    #############

    # Transformations
    transform = T.Compose([
        T.Resize(config['DATASET_VAL']['RESIZE']),
        T.ToTensor(),
        T.Normalize(mean=config['DATASET_VAL']['MEAN'],
                    std=config['DATASET_VAL']['STD'])
    ])

    ds_name = os.path.basename(args.data_paths[0]).split('_crops')[0]
    coo2meter = config['MAX_DIST'][ds_name]
    with open(args.data_paths[0], "rb") as f:
        ds = pickle.load(f)

    test_ds = SceneDataset(ds,
                           coo2meter,
                           feature_model,
                           device,
                           transform)

    # Model Definition
    node_feature_dim = test_ds[0]['node_embeds'].shape[1]
    model = LANDER(feature_dim=node_feature_dim,
                   nhid=args.hidden,
                   num_conv=args.num_conv,
                   dropout=args.dropout,
                   use_GAT=args.gat,
                   K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat)

    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    rand_index = []
    ami = []
    homogeneity = []
    completeness = []
    v_measure = []

    for sample in test_ds:
        labels = sample['node_labels']
        predictions = inference(sample['node_embeds'], labels.copy(), sample['xws'], sample['yws'], coo2meter, sample['cam_ids'], model, device, args)
        # print(f'lab: {labels}')
        # print(f'pred: {predictions}')
        # print(f'ari: {metrics.ari(labels, predictions)}',
        #       f'ami: {metrics.ami(labels, predictions)}',
        #       f'v_mes: {metrics.v_mesure(labels, predictions)}\n')

        rand_index.append(metrics.ari(labels, predictions))
        ami.append(metrics.ami(labels, predictions))
        homogeneity.append(metrics.homg_score(labels, predictions))
        completeness.append(metrics.cmplts_score(labels, predictions))
        v_measure.append(metrics.v_mesure(labels, predictions))

    m_ridx = np.mean(np.asarray(rand_index))
    m_ami = np.mean(np.asarray(ami))
    m_hom = np.mean(np.asarray(homogeneity))
    m_cmplt = np.mean(np.asarray(completeness))
    m_vmes = np.mean(np.asarray(v_measure))

    print(f'Rand index mean = {m_ridx}')
    print(f'Mutual index mean = {m_ami}')
    print(f'homogeneity mean = {m_hom}')
    print(f'completeness mean = {m_cmplt}')
    print(f'v_measure mean = {m_vmes}')
    print('\n')

if __name__== '__main__':
    mp.set_start_method('spawn')
    ###########
    # ArgParser
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--data_paths", type=str, nargs='+', required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for multiprocessing load, 0 for mp to be off')
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--faiss_gpu", action="store_true")

    # KNN
    parser.add_argument("--knn_k", type=int, default=10)

    # Model
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num_conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gat", action="store_true")
    parser.add_argument("--gat_k", type=int, default=1)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--use_cluster_feat", action="store_true")

    # Validation
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--threshold", type=str, default="prob")
    parser.add_argument("--early_stop", action="store_true")

    args = parser.parse_args()

    ###########################
    # Environment Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main(args, device)