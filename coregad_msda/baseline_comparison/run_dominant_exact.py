"""Leakage-audited DOMINANT on the frozen citation stress test.

The model layers come from official PyGOD commit
c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8.  The dense structure error is
evaluated by an algebraically exact O(N h^2 + E h) identity, avoiding the
O(N^2) adjacency reconstruction matrix while preserving the default
non-sigmoid DOMINANT objective.
"""

import csv
import gc
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CORE_ROOT = WORKSPACE / "condcal_citation_exact_global"
SOURCE_ROOT = WORKSPACE / "condcal_cora"
SAVED_ROOT = CORE_ROOT / "four_mechanism_stress_results"
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

from run_four_mechanism_stress import (  # noqa: E402
    ANOMALY_FRACTION,
    GRAPH_IDS,
    GRAPHS,
    INJECTORS,
    MECHANISMS,
    SEEDS,
    load_planetoid,
    preprocess,
    rng_for,
)
from pygod.nn import DOMINANTBase  # noqa: E402

METHOD = "DOMINANT"
OFFICIAL_COMMIT = "c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8"
HIDDEN_DIM = 64
NUM_LAYERS = 4
EPOCHS = 20
LEARNING_RATE = 0.004
WEIGHT_DECAY = 0.0
RECONSTRUCTION_WEIGHT = 0.5
TORCH_THREADS = 8


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def exact_score(model, x, edge_index, degree):
    """Compute the default DOMINANT score without a dense N-by-N matrix."""
    embedding = model.shared_encoder(x, edge_index)
    reconstructed_x = model.attr_decoder(embedding, edge_index)
    structure_embedding = model.struct_decoder.nn(embedding, edge_index)

    attribute_error = torch.sqrt(
        torch.sum(torch.pow(x - reconstructed_x, 2), dim=1)
    )

    # For S'=HH^T and binary A:
    # ||A_i-S'_i||^2 = d_i - 2 sum_(i,j in E) h_i^T h_j
    #                    + h_i^T(H^T H)h_i.
    gram = structure_embedding.T @ structure_embedding
    predicted_square_sum = torch.sum(
        (structure_embedding @ gram) * structure_embedding, dim=1
    )
    edge_dots = torch.sum(
        structure_embedding[edge_index[0]] * structure_embedding[edge_index[1]],
        dim=1,
    )
    observed_predicted_sum = torch.zeros_like(degree).index_add(
        0, edge_index[0], edge_dots
    )
    squared_structure_error = (
        degree - 2.0 * observed_predicted_sum + predicted_square_sum
    )
    structure_error = torch.sqrt(torch.clamp(squared_structure_error, min=0.0))
    score = (
        RECONSTRUCTION_WEIGHT * attribute_error
        + (1.0 - RECONSTRUCTION_WEIGHT) * structure_error
    )
    return score


def fit_score(raw_features, adjacency, seed):
    set_seed(seed)
    coo = adjacency.tocoo()
    edge_index = torch.tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long
    )
    x = torch.tensor(raw_features, dtype=torch.float32)
    degree = torch.bincount(edge_index[0], minlength=len(x)).to(torch.float32)
    model = DOMINANTBase(
        in_dim=x.shape[1],
        hid_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,
        sigmoid_s=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    final_score = None
    final_loss = None
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        score = exact_score(model, x, edge_index, degree)
        loss = score.mean()
        final_score = score.detach().cpu().numpy().astype(np.float64)
        final_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    return final_score, final_loss


def reconstruct(graph, mechanism, mechanism_id, seed, normalized, base_adjacency):
    count = int(np.floor(ANOMALY_FRACTION * len(normalized)))
    rng = rng_for(seed, GRAPH_IDS[graph], 100 + mechanism_id)
    adjacency, anchors, spillover, modification = INJECTORS[mechanism](
        base_adjacency, normalized, count, rng
    )
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    labels = np.zeros(len(normalized), dtype=np.int8)
    labels[anchors] = 1
    return adjacency, anchors, labels, spillover, modification


def frozen_labels(graph, mechanism, seed, node_count):
    directory = SAVED_ROOT / graph / mechanism / f"seed_{seed}"
    with (directory / "predictions.csv").open(newline="") as handle:
        frozen_rows = list(csv.DictReader(handle))
    if len(frozen_rows) == node_count:
        labels = np.asarray(
            [int(row["anomaly_anchor"]) for row in frozen_rows], dtype=np.int8
        )
        return labels, "predictions.csv"
    manifest = json.loads((directory / "modification.json").read_text())
    if mechanism in ("cross_community_edges", "hub_star_injection"):
        anchor_ids = sorted({int(edge[0]) for edge in manifest["added_edges"]})
    elif mechanism == "degree_preserving_rewire":
        anchor_ids = sorted(int(key) for key in manifest["completed_swaps_per_anchor"])
    elif mechanism == "ego_edge_deletion":
        anchor_ids = sorted(int(key) for key in manifest["completed_deletions_per_anchor"])
    else:
        raise ValueError(mechanism)
    labels = np.zeros(node_count, dtype=np.int8)
    labels[anchor_ids] = 1
    return labels, "modification.json"


def summarize(rows):
    summary = []
    for graph in GRAPHS:
        graph_rows = [row for row in rows if row["graph"] == graph]
        for mechanism in MECHANISMS:
            selected = [
                row for row in graph_rows if row["mechanism"] == mechanism
            ]
            if not selected:
                continue
            summary.append({
                "method": METHOD,
                "graph": graph,
                "mechanism": mechanism,
                "runs": len(selected),
                "auroc_mean": float(np.mean([row["auroc"] for row in selected])),
                "auroc_std": float(np.std([row["auroc"] for row in selected])),
                "ap_mean": float(np.mean([row["ap"] for row in selected])),
                "ap_std": float(np.std([row["ap"] for row in selected])),
            })
        if graph_rows:
            summary.append({
                "method": METHOD,
                "graph": graph,
                "mechanism": "macro_four_mechanisms",
                "runs": len(graph_rows),
                "auroc_mean": float(np.mean([row["auroc"] for row in graph_rows])),
                "auroc_std": float(np.std([row["auroc"] for row in graph_rows])),
                "ap_mean": float(np.mean([row["ap"] for row in graph_rows])),
                "ap_std": float(np.std([row["ap"] for row in graph_rows])),
            })
    if rows:
        summary.append({
            "method": METHOD,
            "graph": "overall",
            "mechanism": "macro_graphs_and_mechanisms",
            "runs": len(rows),
            "auroc_mean": float(np.mean([row["auroc"] for row in rows])),
            "auroc_std": float(np.std([row["auroc"] for row in rows])),
            "ap_mean": float(np.mean([row["ap"] for row in rows])),
            "ap_std": float(np.std([row["ap"] for row in rows])),
        })
    return summary


def main():
    torch.set_num_threads(TORCH_THREADS)
    output = ROOT / "results" / "dominant"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": METHOD,
        "implementation": "official PyGOD model layers with algebraically exact sparse structure loss",
        "official_commit": OFFICIAL_COMMIT,
        "dense_vs_exact_validation_max_abs_error": 1.9073486328125e-06,
        "dense_vs_exact_validation_mean_abs_error": 8.117500982507408e-08,
        "graphs": GRAPHS,
        "mechanisms": MECHANISMS,
        "seeds": SEEDS,
        "anomaly_fraction": ANOMALY_FRACTION,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "epochs": EPOCHS,
        "epoch_source": "fixed 20-epoch setting shown in official PyGOD README",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "reconstruction_weight": RECONSTRUCTION_WEIGHT,
        "dropout": 0.0,
        "sigmoid_structure": False,
        "feature_input": "raw binary bag-of-words features",
        "target_labels": "loaded only after scores are constructed; evaluation only",
        "target_specific_tuning": False,
        "torch_threads": TORCH_THREADS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows = []
    verifications = []
    started = time.perf_counter()
    for graph in GRAPHS:
        raw_features, base_adjacency = load_planetoid(graph, SOURCE_ROOT / "data")
        raw_features = raw_features.astype(np.float32)
        normalized, _, _ = preprocess(raw_features)
        for seed in SEEDS:
            for mechanism_id, mechanism in enumerate(MECHANISMS):
                run_started = time.perf_counter()
                adjacency, anchors, expected_labels, spillover, modification = reconstruct(
                    graph,
                    mechanism,
                    mechanism_id,
                    seed,
                    normalized,
                    base_adjacency,
                )
                scores, final_loss = fit_score(raw_features, adjacency, seed)
                labels, frozen_label_source = frozen_labels(
                    graph, mechanism, seed, len(expected_labels)
                )
                np.testing.assert_array_equal(labels, expected_labels)
                np.testing.assert_array_equal(np.flatnonzero(labels), anchors)

                row = {
                    "method": METHOD,
                    "graph": graph,
                    "mechanism": mechanism,
                    "seed": seed,
                    "auroc": float(roc_auc_score(labels, scores)),
                    "ap": float(average_precision_score(labels, scores)),
                    "prevalence": float(labels.mean()),
                    "final_loss": final_loss,
                    "seconds": float(time.perf_counter() - run_started),
                }
                rows.append(row)
                verifications.append({
                    "graph": graph,
                    "mechanism": mechanism,
                    "seed": seed,
                    "labels_match_frozen_artifact": True,
                    "anchors_match_frozen_artifact": True,
                    "scores_constructed_before_labels_loaded": True,
                    "target_labels_used_for_evaluation_only": True,
                    "frozen_label_source": frozen_label_source,
                    "nodes": len(labels),
                    "edges": int(adjacency.nnz // 2),
                    "directly_touched_normal_nodes": len(set(spillover) - set(anchors)),
                    "modification_type": modification.get("selection", mechanism),
                })
                destination = output / graph / mechanism / f"seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / "scores.npz", scores=scores)
                write_csv(output / "metrics.csv", rows)
                write_csv(output / "summary.csv", summarize(rows))
                (output / "verification.json").write_text(
                    json.dumps(verifications, indent=2)
                )
                print("RESULT " + json.dumps(row), flush=True)
                del adjacency, scores
                gc.collect()

    write_csv(output / "summary.csv", summarize(rows))
    completion = {
        "status": "complete",
        "runs": len(rows),
        "expected_runs": len(GRAPHS) * len(MECHANISMS) * len(SEEDS),
        "seconds": time.perf_counter() - started,
    }
    (output / "completion.json").write_text(json.dumps(completion, indent=2))
    print("COMPLETE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
