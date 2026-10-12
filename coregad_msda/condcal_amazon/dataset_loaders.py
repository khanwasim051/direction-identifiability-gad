"""Verified Reddit and Amazon loaders for the final manuscript protocol."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat


def clean_adjacency(edge_index, node_count):
    row, col = edge_index
    keep = row != col
    adjacency = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (row[keep], col[keep])),
        shape=(node_count, node_count),
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    adjacency.sort_indices()
    return adjacency


def read_tensor_archive(path: Path, prefix, specifications):
    """Read simple contiguous tensors from the verified PyTorch ZIP archive."""
    arrays = []
    with zipfile.ZipFile(path) as archive:
        for index, (shape, dtype) in enumerate(specifications):
            raw = archive.read(f"{prefix}/data/{index}")
            expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if len(raw) != expected:
                raise ValueError((path, index, len(raw), expected))
            arrays.append(np.frombuffer(raw, dtype=dtype).reshape(shape).copy())
    return arrays


def load_reddit(data_dir):
    data_dir = Path(data_dir)
    features, edges, labels = read_tensor_archive(
        data_dir / "reddit.pt",
        "archive",
        [
            ((10984, 64), np.float32),
            ((2, 168016), np.int64),
            ((10984,), np.float64),
        ],
    )
    labels = labels.astype(np.int8)
    assert tuple(np.bincount(labels, minlength=2)) == (10618, 366)
    mask = np.ones(len(features), dtype=bool)
    return features, clean_adjacency(edges, len(features)), labels, mask


def load_amazon(data_dir):
    data_dir = Path(data_dir)
    data = loadmat(data_dir / "amazon_raw" / "Amazon.mat")
    features = np.asarray(data["features"].todense(), dtype=np.float32)
    labels = data["label"].ravel().astype(np.int8)
    adjacency = data["net_upu"].tocsr().maximum(data["net_upu"].T.tocsr())
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    adjacency.sort_indices()
    evaluation_mask = np.arange(len(labels)) >= 3305
    assert features.shape == (11944, 25)
    assert tuple(np.bincount(labels[evaluation_mask], minlength=2)) == (7818, 821)
    assert adjacency.nnz // 2 == 175608
    return features, adjacency, labels, evaluation_mask
