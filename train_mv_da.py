import argparse
import os
import glob
import yaml
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms as T
from torchvision.transforms import (
    ColorJitter, RandomHorizontalFlip
)
from torch.utils.tensorboard import SummaryWriter
from dataset import SceneDataset, GraphDataset, prepare_dataset_graphs_mp
from models import LANDER, load_feature_extractor
from utils import build_next_level, decode, stop_iterating, l2norm, metrics

from test_mv_da import inference

torch.manual_seed(123)

def collate(batch):
    graphs = []
    for sample in batch:
        graphs.extend([g for g in sample['graphs']])
    return graphs

def main(args, device, collate_fun):
    # Load config
    config = yaml.safe_load(open('config/config_training.yaml', 'r'))

    #################################
    # Load feature extraction model #
    #################################
    feature_model = load_feature_extractor(config, device)

    #############
    # Load data #
    #############
    feature_dim = 256

    # Transformations
    val_transform = T.Compose([
        T.Resize(config['DATASET_VAL']['RESIZE']),
        T.ToTensor(),
        T.Normalize(mean=config['DATASET_VAL']['MEAN'],
                    std=config['DATASET_VAL']['STD'])
    ])

    train_transform = T.Compose([
        T.Resize(config['DATASET_VAL']['RESIZE']),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0),
        T.ToTensor(),
        T.Normalize(mean=config['DATASET_VAL']['MEAN'],
                    std=config['DATASET_VAL']['STD'])
    ])

    train_ds = []
    val_ds = []
    for data_path in args.data_paths:
        with open(data_path, "rb") as f:
            ds = pickle.load(f)

        split_idx = int(0.85 * len(ds))
        train_seq = ds[:split_idx]
        val_seq = ds[split_idx:]

        train_ds.append(SceneDataset(train_seq,
                                     feature_dim,
                                     feature_model,
                                     device,
                                     train_transform))
        val_ds.append(SceneDataset(val_seq,
                                   feature_dim,
                                   feature_model,
                                   device,
                                   val_transform))

    train_ds = torch.utils.data.ConcatDataset(train_ds)
    val_ds = torch.utils.data.ConcatDataset(val_ds)

    train_len = len(train_ds) * args.batch_size
    val_len = len(val_ds) * args.batch_size

    # Final validation ds can be created once
    scene_seq = [sample for sample in val_ds]
    gss = prepare_dataset_graphs_mp(sequence=scene_seq,
                                    k=args.knn_k,
                                    levels=args.levels,
                                    faiss_gpu=args.faiss_gpu,
                                    device=device,
                                    num_workers=config['DATALOADER']['NUM_WORKERS'])
    val_gs_ds = GraphDataset(scene_seq, gss)
    val_gs_dl = torch.utils.data.DataLoader(dataset=val_gs_ds,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            collate_fn=collate_fun,
                                            drop_last=False)

    ##################
    # Model Definition
    node_feature_dim = 2 * feature_dim
    model = LANDER(feature_dim=node_feature_dim,
                   nhid=args.hidden,
                   num_conv=args.num_conv,
                   dropout=args.dropout,
                   use_GAT=args.gat,
                   K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat,
                   use_focal_loss=args.use_focal_loss)
    model = model.to(device)
    model.train()
    best_vmes = -np.Inf

    #################
    # Hyperparameters
    opt = optim.Adam(
        model.parameters(),
        lr=1e-6,
        # momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CyclicLR(
        opt,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        mode='triangular2',
        step_size_up=args.epochs // 10,
        cycle_momentum=False
    )

    #################
    # Checkpoint paths
    model_dir = os.path.join(args.checkpoint_dir, args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tb_writer = SummaryWriter(log_dir=model_dir)

    ###############
    # Training Loop
    for epoch in range(args.epochs):
        all_loss_den = 0
        all_loss_conn = 0
        model.train()

        scene_seq = [sample for sample in train_ds]
        gss = prepare_dataset_graphs_mp(sequence=scene_seq,
                                        k=args.knn_k,
                                        levels=args.levels,
                                        faiss_gpu=args.faiss_gpu,
                                        device=device,
                                        num_workers=config['DATALOADER']['NUM_WORKERS'])
        train_gs_ds = GraphDataset(scene_seq, gss)
        train_gs_dl = torch.utils.data.DataLoader(dataset=train_gs_ds,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_fun,
                                                  drop_last=False)

        for batch in train_gs_dl:
            loss = 0
            opt.zero_grad()
            for g in batch:
                processed_g = model(g)
                curr_loss, loss_den, loss_conn = model.compute_loss(processed_g)
                all_loss_den += loss_den
                all_loss_conn += loss_conn
                loss += curr_loss
            loss /= args.batch_size
            loss.backward()
            opt.step()
        scheduler.step()

        # Record
        avg_loss_den = all_loss_den / train_len
        avg_loss_conn = all_loss_conn / train_len
        print(
            "Training, epoch: %d, loss_den: %.6f, loss_conn: %.6f"
            % (epoch, avg_loss_den, avg_loss_conn)
        )

        # Add to tensorboard
        tb_writer.add_scalar('Train/Den_loss', avg_loss_den, epoch)
        tb_writer.add_scalar('Train/Conn_loss', avg_loss_conn, epoch)

        # Report validation
        all_loss_den_val = 0
        all_loss_conn_val = 0

        rand_index = []
        ami = []
        homogeneity = []
        completeness = []
        v_measure = []
        with torch.no_grad():
            for val_batch in val_gs_dl:
                for g in val_batch:
                    processed_g = model(g)
                    _, loss_den_val, loss_conn_val = model.compute_loss(processed_g)
                    all_loss_den_val += loss_den_val
                    all_loss_conn_val += loss_conn_val

        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

        avg_loss_den_val = all_loss_den_val / val_len
        avg_loss_conn_val = all_loss_conn_val / val_len
        
        # Record
        print(
            "Validation, epoch: %d, loss_den: %.6f, loss_conn: %.6f"
            % (epoch, avg_loss_den_val, avg_loss_conn_val)
        )
        # Add to tensorboard
        tb_writer.add_scalar('Val_Loss/Den_loss', avg_loss_den_val, epoch)
        tb_writer.add_scalar('Val_Loss/Conn_loss', avg_loss_conn_val, epoch)

        # Validation metrics
        model.eval()
        for sample in val_ds:
            labels = sample['node_labels']
            predictions = inference(sample['node_embeds'], labels.copy(), sample['xws'], sample['yws'], sample['cam_ids'], model, device, args)

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

        # Save ckpt
        if m_vmes > best_vmes:
            best_vmes = m_vmes
            print("\nNew best epoch", epoch)
            best_model = glob.glob(os.path.join(model_dir, 'model_best-*.pth'))
            if len(best_model):
                os.remove(best_model[0])
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_best-{epoch}.pth'))

        print(f'Rand index mean = {m_ridx}')
        print(f'Mutual index mean = {m_ami}')
        print(f'homogeneity mean = {m_hom}')
        print(f'completeness mean = {m_cmplt}')
        print(f'v_measure mean = {m_vmes}')
        print('\n')

        tb_writer.add_scalar('Val_Metrics/Rand_idx', m_ridx, epoch)
        tb_writer.add_scalar('Val_Metrics/AMI', m_ami, epoch)
        tb_writer.add_scalar('Val_Metrics/Homogeneity', m_hom, epoch)
        tb_writer.add_scalar('Val_Metrics/Completeness', m_cmplt, epoch)
        tb_writer.add_scalar('Val_Metrics/V_Measure', m_vmes, epoch)

if __name__== '__main__':
    mp.set_start_method('spawn')
    ###########
    # ArgParser
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--data_paths", type=str, nargs='+', required=True)
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers for multiprocessing load, 0 for mp to be off')
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--model_name", type=str, default="model_001")

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
    parser.add_argument("--use_focal_loss", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=0.1)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=0.)

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

    main(args, device, collate)