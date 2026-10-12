"""Equation-faithful sparse AnomalyGFM transfer to Reddit and Amazon.

The released zero-shot implementation densifies each adjacency matrix.  This
refactor preserves its two-layer GCN, source-supervised prototype-alignment
loss, hyperparameters, and final ``abnormal + 6 * normal`` score, but performs
graph propagation with sparse matrices.  Training uses only the official
feature-aligned Facebook source graph.  Target labels are withheld until after
both target score vectors have been constructed.

This is an independent sparse refactor of the official code, not an official
pretrained checkpoint result.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXTERNAL_DATA_ROOT = Path(
    os.environ.get("COREGAD_EXTERNAL_DATA", WORKSPACE / "external_data")
)
sys.path.insert(0, str(WORKSPACE / "condcal_amazon"))
from dataset_loaders import load_amazon, load_reddit  # noqa: E402


SEEDS = [11, 23, 37, 53, 71]
EPOCHS = 301
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 400
INPUT_DIM = 8
TRAIN_RATE = 0.3
VALIDATION_RATE = 0.1
MID_NORMAL_WEIGHT = 6.0
TORCH_THREADS = 8


def clean_adjacency(matrix):
    matrix = sparse.csr_matrix(matrix).maximum(sparse.csr_matrix(matrix).T)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.data[:] = 1
    matrix.sort_indices()
    return matrix


def row_normalize(features):
    features = np.asarray(features, dtype=np.float32)
    sums = features.sum(axis=1, keepdims=True)
    return np.divide(features, sums, out=np.zeros_like(features), where=sums != 0)


def rank8_svd(features):
    features = np.asarray(features, dtype=np.float32)
    u, singular_values, _ = np.linalg.svd(features, full_matrices=False)
    return (u[:, :INPUT_DIM] * singular_values[:INPUT_DIM]).astype(np.float32)


def scipy_to_torch(matrix):
    coo = sparse.coo_matrix(matrix)
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def normalized_adjacencies(adjacency):
    adjacency = clean_adjacency(adjacency).astype(np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inverse_sqrt[positive] = degree[positive] ** -0.5
    diagonal = sparse.diags(inverse_sqrt, format="csr")
    normalized = diagonal @ adjacency @ diagonal
    normalized = normalized + sparse.eye(len(degree), dtype=np.float32, format="csr")
    inverse = np.zeros_like(degree)
    inverse[positive] = 1.0 / degree[positive]
    neighbor_mean = sparse.diags(inverse, format="csr") @ adjacency
    return scipy_to_torch(normalized), scipy_to_torch(neighbor_mean)


class SparseGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.PReLU()
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features, adjacency):
        return self.activation(torch.sparse.mm(adjacency, self.linear(features)) + self.bias)


class SparseAnomalyGFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = SparseGCN(INPUT_DIM, EMBEDDING_DIM)
        self.gcn2 = SparseGCN(EMBEDDING_DIM, EMBEDDING_DIM)
        self.logit = nn.Linear(EMBEDDING_DIM, 1, bias=False)
        self.residual_logit = nn.Linear(EMBEDDING_DIM, 1, bias=False)
        self.normal_prompt = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.abnormal_prompt = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)

    def forward(self, features, normalized_adjacency, neighbor_mean,
                normal_prompt_raw, abnormal_prompt_raw):
        embedding = self.gcn2(self.gcn1(features, normalized_adjacency), normalized_adjacency)
        normal_prompt = F.relu(self.normal_prompt(normal_prompt_raw))
        abnormal_prompt = F.relu(self.abnormal_prompt(abnormal_prompt_raw))
        residual = embedding - torch.sparse.mm(neighbor_mean, embedding)
        return (
            self.logit(embedding).squeeze(1),
            self.residual_logit(residual).squeeze(1),
            residual,
            normal_prompt,
            abnormal_prompt,
        )


def load_source():
    data = loadmat(EXTERNAL_DATA_ROOT / "Facebook_svd.mat")
    features = row_normalize(data["Attributes"])
    labels = np.asarray(data["Label"]).ravel().astype(np.float32)
    adjacency = clean_adjacency(data["Network"])
    return features, adjacency, labels


def load_targets():
    reddit_mat = loadmat(EXTERNAL_DATA_ROOT / "reddit_svd.mat")
    reddit_features = row_normalize(reddit_mat["Attributes"])
    reddit_adjacency = clean_adjacency(reddit_mat["Network"])
    _, _, reddit_labels, reddit_mask = load_reddit(WORKSPACE / "condcal_amazon" / "data")

    amazon_raw, amazon_adjacency, amazon_labels, amazon_mask = load_amazon(
        WORKSPACE / "condcal_amazon" / "data"
    )
    amazon_features = row_normalize(rank8_svd(amazon_raw))
    return {
        "Reddit": (reddit_features, reddit_adjacency, reddit_labels, reddit_mask),
        "Amazon": (amazon_features, amazon_adjacency, amazon_labels, amazon_mask),
    }


def source_split(labels, seed):
    random.seed(seed)
    indices = list(range(len(labels)))
    random.shuffle(indices)
    n_train = int(len(indices) * TRAIN_RATE)
    n_validation = int(len(indices) * VALIDATION_RATE)
    return np.asarray(indices[:n_train]), np.asarray(indices[n_train:n_train + n_validation])


def train_source(seed, features, adjacency, labels):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_ids, validation_ids = source_split(labels, seed)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    normalized, neighbor_mean = normalized_adjacencies(adjacency)
    model = SparseAnomalyGFM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
    model.train()
    final_loss = None
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        normal_raw = torch.randn(EMBEDDING_DIM)
        abnormal_raw = torch.randn(EMBEDDING_DIM)
        logits, _, residual, normal_prompt, abnormal_prompt = model(
            x, normalized, neighbor_mean, normal_raw, abnormal_raw
        )
        classification = criterion(logits[train_ids], y[train_ids])
        residual_train = residual[train_ids]
        labels_train = y[train_ids]
        normal_distance = torch.sqrt(torch.sum(
            (normal_prompt - residual_train[labels_train == 0]) ** 2, dim=1
        )).mean()
        abnormal_distance = torch.sqrt(torch.sum(
            (abnormal_prompt - residual_train[labels_train == 1]) ** 2, dim=1
        )).mean()
        loss = classification + abnormal_distance + 0.1 * normal_distance
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return model, final_loss, train_ids, validation_ids


@torch.no_grad()
def score_target(model, features, adjacency, prompt_seed):
    torch.manual_seed(prompt_seed)
    x = torch.tensor(features, dtype=torch.float32)
    normalized, neighbor_mean = normalized_adjacencies(adjacency)
    model.eval()
    normal_raw = torch.randn(EMBEDDING_DIM)
    abnormal_raw = torch.randn(EMBEDDING_DIM)
    _, _, residual, normal_prompt, abnormal_prompt = model(
        x, normalized, neighbor_mean, normal_raw, abnormal_raw
    )
    residual = F.normalize(residual, p=2, dim=1)
    normal_prompt = F.normalize(normal_prompt, p=2, dim=0)
    abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
    score_normal = residual @ normal_prompt
    score_abnormal = residual @ abnormal_prompt
    return (
        torch.exp(score_abnormal) + MID_NORMAL_WEIGHT * torch.exp(-score_normal)
    ).cpu().numpy().astype(np.float64)


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(TORCH_THREADS)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    source_features, source_adjacency, source_labels = load_source()
    targets = load_targets()
    rows = []
    for seed in SEEDS:
        start = time.perf_counter()
        model, final_loss, train_ids, validation_ids = train_source(
            seed, source_features, source_adjacency, source_labels
        )
        training_seconds = time.perf_counter() - start
        target_scores = {}
        target_times = {}
        for target_id, (name, (features, adjacency, _, _)) in enumerate(targets.items()):
            start = time.perf_counter()
            target_scores[name] = score_target(model, features, adjacency,
                                               100_000 + seed * 10 + target_id)
            target_times[name] = time.perf_counter() - start

        # Only now are target labels and masks used.
        for name, (_, _, labels, mask) in targets.items():
            score = target_scores[name][mask]
            y = labels[mask]
            auroc = float(roc_auc_score(y, score))
            ap = float(average_precision_score(y, score))
            rows.append({
                "method": "AnomalyGFM_sparse_refactor",
                "source": "Facebook_svd",
                "target": name,
                "seed": seed,
                "auroc": auroc,
                "ap": ap,
                "prevalence": float(np.mean(y)),
                "source_training_seconds": training_seconds,
                "target_inference_seconds": target_times[name],
                "final_source_loss": final_loss,
                "source_train_nodes": len(train_ids),
                "source_validation_nodes": len(validation_ids),
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
            })
            print(f"seed={seed} {name} AUROC={auroc:.6f} AP={ap:.6f} "
                  f"train={training_seconds:.2f}s infer={target_times[name]:.2f}s", flush=True)

    write_csv(output / "metrics.csv", rows)
    summary = []
    for name in targets:
        selected = [row for row in rows if row["target"] == name]
        summary.append({
            "method": "AnomalyGFM_sparse_refactor",
            "source": "Facebook_svd",
            "target": name,
            "runs": len(selected),
            "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
            "auroc_std": float(np.std([r["auroc"] for r in selected], ddof=0)),
            "ap_mean": float(np.mean([r["ap"] for r in selected])),
            "ap_std": float(np.std([r["ap"] for r in selected], ddof=0)),
            "training_seconds_mean": float(np.mean([r["source_training_seconds"] for r in selected])),
            "inference_seconds_mean": float(np.mean([r["target_inference_seconds"] for r in selected])),
        })
    write_csv(output / "summary.csv", summary)
    protocol = {
        "status": "completed independent equation-faithful sparse refactor",
        "official_repository": "https://github.com/mala-lab/AnomalyGFM",
        "source": "official Facebook_svd.mat aligned source graph",
        "targets": ["Reddit", "Amazon"],
        "weibo_excluded": True,
        "amazon_graph": "same 11944-node U-P-U graph and 8639-node evaluation mask used by all other methods",
        "target_feature_alignment": "official aligned Reddit features; rank-8 U Sigma SVD for Amazon",
        "source_labels_used": True,
        "target_labels_used_for_scoring_or_tuning": False,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "final_score": "exp(abnormal cosine) + 6 * exp(-normal cosine)",
        "dense_to_sparse_change": "algebraically identical adjacency propagation and neighbor mean",
        "seeds": SEEDS,
        "torch_threads": TORCH_THREADS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output / "completion.json").write_text(json.dumps({"complete": True, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
