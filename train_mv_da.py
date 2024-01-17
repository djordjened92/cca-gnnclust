import argparse
import os
import glob
import math
import yaml
import pickle
import random
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms as T
from torchvision.transforms import (
    ColorJitter, RandomHorizontalFlip, RandomRotation
)
from torch.utils.tensorboard import SummaryWriter
from dataset import SceneDataset, GraphDataset, prepare_dataset_graphs_mp
from models import LANDER, load_feature_extractor
from utils import build_next_level, decode, stop_iterating, l2norm, metrics

from test_mv_da import inference

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.2,
        r1=0.3,
        mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

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

    # Transformations
    val_transform = T.Compose([
        T.Resize(config['DATASET_VAL']['RESIZE']),
        T.ToTensor(),
        T.Normalize(mean=config['DATASET_VAL']['MEAN'],
                    std=config['DATASET_VAL']['STD'])
    ])

    train_transform = T.Compose([
        T.Resize(config['DATASET_TRAIN']['RESIZE']),
        RandomRotation(20),
        ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0),
        T.ToTensor(),
        T.Normalize(mean=config['DATASET_TRAIN']['MEAN'],
                    std=config['DATASET_TRAIN']['STD']),
        RandomErasing(probability=0.3, mean=config['DATASET_TRAIN']['MEAN'])
    ])

    data_root = config['DATA_ROOT']

    # Load training
    train_ds = []
    for d in config['DATASET_TRAIN']['NAME']:
        data_path = os.path.join(data_root, d + '_crops.pkl')
        coo2meter = config['MAX_DIST'][d]
        with open(data_path, "rb") as f:
            train_seq = pickle.load(f)

        train_ds.append(SceneDataset(train_seq,
                                     coo2meter,
                                     feature_model,
                                     device,
                                     val_transform))
    train_ds = torch.utils.data.ConcatDataset(train_ds)
    print(f'Training length: {len(train_ds)}')

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

    # Load validation
    data_path = os.path.join(data_root, config['DATASET_VAL']['NAME'] + '_crops.pkl')
    coo2meter = config['MAX_DIST'][d]
    with open(data_path, "rb") as f:
        val_seq = pickle.load(f)
    val_ds = SceneDataset(val_seq,
                          coo2meter,
                          feature_model,
                          device,
                          val_transform)
    print(f'Validation length: {len(val_ds)}')

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
    node_feature_dim = train_ds[0]['node_embeds'].shape[1]
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
        lr=args.base_lr,
        # momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     opt,
    #     T_max=args.epochs
    # )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.base_lr,
        steps_per_epoch=math.ceil(len(train_ds) / args.batch_size),
        epochs=args.epochs
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

        for batch in train_gs_dl:
            loss = 0
            opt.zero_grad()
            curr_loss_den = 0
            curr_loss_conn = 0
            for g in batch:
                processed_g = model(g)
                loss_supcon, loss_conn = model.compute_loss(processed_g)
                conn_losses = loss_conn.mean()
                supcon_losses = loss_supcon.mean()
                loss = supcon_losses + conn_losses

                curr_loss_den += supcon_losses.item()
                curr_loss_conn += conn_losses.item()

            loss /= args.batch_size
            all_loss_conn += curr_loss_conn / args.batch_size
            all_loss_den += curr_loss_den / args.batch_size
            loss.backward()
            opt.step()
        scheduler.step()

        # Record
        avg_loss_den = all_loss_den / len(train_gs_dl)
        avg_loss_conn = all_loss_conn / len(train_gs_dl)
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

        with torch.no_grad():
            for val_batch in val_gs_dl:
                curr_loss_den = 0
                curr_loss_conn = 0
                for g in val_batch:
                    processed_g = model(g)
                    loss_supcon, loss_conn = model.compute_loss(processed_g)
                    curr_loss_den += loss_supcon.mean().item()
                    curr_loss_conn += loss_conn.mean().item()
                all_loss_den_val += curr_loss_den / args.batch_size
                all_loss_conn_val += curr_loss_conn / args.batch_size

        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

        avg_loss_den_val = all_loss_den_val / len(val_gs_dl)
        avg_loss_conn_val = all_loss_conn_val / len(val_gs_dl)
        
        # Record
        print(
            "Validation, epoch: %d, loss_den: %.6f, loss_conn: %.6f"
            % (epoch, avg_loss_den_val, avg_loss_conn_val)
        )
        # Add to tensorboard
        tb_writer.add_scalar('Val_Loss/Den_loss', avg_loss_den_val, epoch)
        tb_writer.add_scalar('Val_Loss/Conn_loss', avg_loss_conn_val, epoch)

        # Validation metrics
        rand_index = []
        ami = []
        homogeneity = []
        completeness = []
        v_measure = []
        if epoch % 5 == 0:
            model.eval()
            for sample in val_ds:
                labels = sample['node_labels']
                predictions = inference(sample['node_embeds'],
                                        labels.copy(),
                                        sample['xws'],
                                        sample['yws'],
                                        coo2meter,
                                        sample['cam_ids'],
                                        model,
                                        device,
                                        args)

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