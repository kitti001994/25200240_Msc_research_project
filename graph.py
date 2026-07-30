"""
Builds the functional connectivity graph from parcel time-series.

The Brainnetome atlas constructs graph edges from Pearson correlation between parcel BOLD signals,
thresholded at |r| > CORR_THRESHOLD.

Two edge construction modes:
  - population_graph : computed once from the training-set average
  - subject_graph : computed per subject (for inference)

The ST-GCN model uses fixed population-level graph during training.
"""

import os
import numpy as np
import torch
from scipy.stats import pearsonr

import config


def compute_correlation_matrix(time_series_list: list):
    """
    List of (T, N) BOLD arrays (one per run per subject),
    compute the average Pearson correlation matrix (N, N).
    """
    if not time_series_list:
        raise ValueError(
            "time_series_list is empty. No BOLD files were found. "
            "Check fMRIPrep output path."
        )
    N = time_series_list[0].shape[1]
    cumulative = np.zeros((N, N))
    count = 0

    for ts in time_series_list:
        # Normalise each column
        ts_normed = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-8)
        corr = (ts_normed.T @ ts_normed) / ts.shape[0]
        cumulative += corr
        count += 1

    return cumulative / count # (N, N)


def threshold_graph(
    corr_matrix: np.ndarray,
    threshold: float = config.CORR_THRESHOLD,
    add_self_loops: bool = config.SELF_LOOPS,
):
    """
    Threshold the correlation matrix and return
    (edge_index, edge_weight) tensors.
    """
    N = corr_matrix.shape[0]
    adj = (np.abs(corr_matrix) > threshold).astype(np.float32)
    adj *= corr_matrix # signed weights

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)

    src, dst = np.where(adj != 0)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)

    print(f"  Graph: {N} nodes, {edge_index.shape[1]} edges "
          f"(threshold={threshold}, self_loops={add_self_loops})")
    return edge_index, edge_weight


# Population-level graph (computed 1x from training subjects)
def build_population_graph(
    time_series_list: list,
    cache_path: str = None,
):
    """
    Population-level graph and cache
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        edge_index  = torch.tensor(data["edge_index"],  dtype=torch.long)
        edge_weight = torch.tensor(data["edge_weight"], dtype=torch.float32)
        n_nodes = int(data["n_nodes"][0]) if "n_nodes" in data else int(edge_index.max()) + 1
        print(f"Loaded population graph from {cache_path} ({n_nodes} nodes)")
        return edge_index, edge_weight

    print("Building population-level functional connectivity graph.")
    corr = compute_correlation_matrix(time_series_list)
    edge_index, edge_weight = threshold_graph(corr)

    if cache_path:
        cache_parent = os.path.dirname(cache_path)
        if cache_parent:
            os.makedirs(cache_parent, exist_ok=True)
        np.savez_compressed(
            cache_path,
            edge_index=edge_index.numpy(),
            edge_weight=edge_weight.numpy(),
            n_nodes=np.array([time_series_list[0].shape[1]], dtype=np.int32),
            threshold=np.array([config.CORR_THRESHOLD], dtype=np.float32),
            self_loops=np.array([int(config.SELF_LOOPS)], dtype=np.int8),
        )
        print(f"Saved population graph to {cache_path}")

    return edge_index, edge_weight


# Degree-normalised adjacency
def to_dense_adj(edge_index, edge_weight, N):
    """Convert edge_index / edge_weight back to dense (N, N) matrix."""
    A = torch.zeros(N, N)
    A[edge_index[0], edge_index[1]] = edge_weight
    return A
