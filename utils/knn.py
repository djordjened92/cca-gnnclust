#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import math
import multiprocessing as mp
import os
import faiss

import numpy as np
from tqdm import tqdm
from utils import Timer, l2norm

from .faiss_search import faiss_search_knn
from scipy.sparse import csr_matrix

__all__ = [
    "knn_faiss",
    "fast_knns2spmat",
    "build_knns",
    "knns2ordered_nbrs",
    "mark_same_camera_nbrs",
    "build_knn_per_camera"
]

np.seterr(divide='ignore')

def build_knn_per_camera(features, cam_ids, k=3):
    num_of_nodes = len(cam_ids)
    cam_orders = np.argsort(cam_ids)

    vals, counts = np.unique(cam_ids[cam_orders], return_counts=True)

    feat_sorted = features[cam_orders]
    sims = feat_sorted @ feat_sorted.T

    cnt_cumsum = np.cumsum(np.concatenate(([0], counts)), axis=0)
    split_edges = np.cumsum(counts)[:-1]

    vert_splits = np.split(sims, split_edges)

    # Initialize empty similarities and neighbours matrix
    all_sims = np.empty((num_of_nodes, num_of_nodes * k))
    all_sims.fill(0.)
    all_idcs = np.empty((num_of_nodes, num_of_nodes * k))
    all_idcs.fill(-1)

    for i, v in enumerate(vert_splits):
        hor_split = np.split(v, split_edges, axis=1)
        part_idcs = []
        part_sims = []
        for j, part in enumerate(hor_split):
            if i != j:
                # Check if no enough neighbours
                if part.shape[1] == 1:
                    idcs = np.zeros_like(part, dtype=int)
                    max_vals = part
                else:
                    k_curr = min(k, part.shape[1] - 1)
                    idcs = np.argpartition(part, -k_curr)[..., -k_curr:]
                    max_vals = np.take_along_axis(part, idcs, 1)

                idcs += cnt_cumsum[j]
                idcs = cam_orders[idcs]
                part_idcs.append(idcs)
                part_sims.append(max_vals)
        
        if len(part_idcs) > 0:
            part_idcs = np.concatenate(part_idcs, axis=1)
            part_sims = np.concatenate(part_sims, axis=1)

            all_idcs[cnt_cumsum[i]:cnt_cumsum[i] + part_idcs.shape[0], :part_idcs.shape[1]] = part_idcs
            all_sims[cnt_cumsum[i]:cnt_cumsum[i] + part_sims.shape[0], :part_sims.shape[1]] = part_sims

    knns = np.concatenate((all_idcs[:, None, :], all_sims[:, None, :]), axis=1)

    remap_fin = ((cam_orders - np.expand_dims(np.arange(len(cam_orders)), 0).T) == 0).nonzero()[1]
    final_knns = knns[remap_fin]

    return final_knns

def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    sims = knns[:, 1, :]
    if sort:
        # sort sims from high to low
        nb_idx = np.argsort(sims, axis=1)[..., ::-1]
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        sims = sims[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return sims, nbrs

def mark_same_camera_nbrs(knns, cam_ids):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)

    # Set nbr to -1 if from same camera
    same_cam = cam_ids[:, None] == cam_ids[nbrs]
    knns[:, 0, :][same_cam] = -1 # neighbours
    knns[:, 1, :][same_cam] = 0. # similarities

    return knns


def fast_knns2spmat(knns, k, th_sim=-1, fill_value=None):
    # convert knns to symmetric sparse matrix
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.zeros([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown sim to 0 and nbr to -1
        for i, (nbr, sim) in enumerate(knns):
            size = len(nbr)
            assert size == len(sim)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = sim[:size]
        knns = ndarr
    nbrs = knns[:, 0, :]
    sims = knns[:, 1, :]

    if fill_value is not None:
        print("[fast_knns2spmat] edge fill value:", fill_value)
        sims.fill(fill_value)

    row, col = np.where(sims >= th_sim)
    # remove the self-loop and same camera instances
    idxs = nbrs[row, col] != -1
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col].astype(int)  # convert to absolute column

    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_knns(feats, k, xws, yws, coo2meter, faiss_gpu):
    index = knn_faiss(feats, k, xws, yws, coo2meter, using_gpu=faiss_gpu)
    knns = index.get_knns()
    return knns


class knn:
    def __init__(self, feats, k, index_path="", verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_sims = []
        nbrs, sims = self.knns[i]
        for n, sim in zip(nbrs, sims):
            if 1 - sim < self.th:
                continue
            th_nbrs.append(n)
            th_sims.append(sim)
        th_nbrs = np.array(th_nbrs)
        th_sims = np.array(th_sims)
        return (th_nbrs, th_sims)

    def get_knns(self, th=None):
        if th is None or th <= 0.0:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer(
            "filter edges by th {} (CPU={})".format(th, nproc), self.verbose
        ):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot)
                )
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    def __init__(
        self,
        feats,
        k,
        xws,
        yws,
        coo2meter,
        using_gpu=True
    ):
        feats = feats.astype("float32")
        size, dim = feats.shape
        dot_prod = feats @ feats.T

        # Expand similarities with position similarities
        coordinates = np.concatenate((xws, yws), axis=1) / coo2meter
        coo_dist = np.linalg.norm(coordinates[:, None, :] - coordinates, axis=-1)
        # dist_thrsh = 1.5
        # scores = np.where(coo_dist <= dist_thrsh, dot_prod, 0.)
        scores = dot_prod

        # Find nearest neighbours
        idcs = np.argpartition(scores, -k)[..., -k:]
        min_scores = np.take_along_axis(scores, idcs, 1)

        self.knns = list(zip(idcs, min_scores))