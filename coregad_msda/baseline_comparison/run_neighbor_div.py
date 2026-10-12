"""Leakage-audited NeighborDiv baseline on the frozen citation stress test.

This is an equation-faithful reproduction of NeighborDiv (arXiv:2605.20879v1)
because no official implementation was linked in the paper at the time this
experiment was frozen.  It uses r=8 and exhaustive unordered neighbor pairs,
so there is no pair-sampling randomness.  Synthetic labels are loaded only
after score construction and are used only for metric computation.
"""

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.extmath import randomized_svd

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

METHOD = "NeighborDiv"
PROJECTION_RANK = 8
SVD_RANDOM_STATE = 0


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def projected_features(raw_features):
    """Return the paper's U_r Sigma_r projection, normalized for cosine."""
    u, singular_values, _ = randomized_svd(
        raw_features,
        n_components=PROJECTION_RANK,
        random_state=SVD_RANDOM_STATE,
    )
    embedding = u * singular_values

    # The paper applies row-wise L1 normalization before cosine similarity.
    l1 = np.abs(embedding).sum(axis=1, keepdims=True)
    embedding = np.divide(
        embedding, l1, out=np.zeros_like(embedding), where=l1 > 0
    )
    l2 = np.linalg.norm(embedding, axis=1, keepdims=True)
    return np.divide(
        embedding, l2, out=np.zeros_like(embedding), where=l2 > 0
    ).astype(np.float32)


def neighbor_div_scores(embedding, adjacency):
    """Compute exhaustive neighbor-pair variance and global calibration."""
    n = adjacency.shape[0]
    diversity = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for node in range(n):
        neighbors = adjacency.indices[
            adjacency.indptr[node] : adjacency.indptr[node + 1]
        ]
        degree = len(neighbors)
        if degree < 2:
            continue
        first, second = np.triu_indices(degree, k=1)
        similarities = np.einsum(
            "ij,ij->i", embedding[neighbors[first]], embedding[neighbors[second]]
        )
        diversity[node] = similarities.var(ddof=0)
        valid[node] = True

    reference = np.median(diversity[valid])
    deviation = np.abs(diversity - reference)
    mean = deviation[valid].mean()
    std = deviation[valid].std(ddof=0)
    scores = np.zeros(n, dtype=np.float64)
    if std > 0:
        scores[valid] = (deviation[valid] - mean) / std
    return scores, diversity, valid, reference


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
    """Load frozen labels, falling back to the intervention manifest."""
    directory = SAVED_ROOT / graph / mechanism / f"seed_{seed}"
    prediction_path = directory / "predictions.csv"
    with prediction_path.open(newline="") as handle:
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
    if not rows:
        return summary
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
    output = ROOT / "results" / "neighbor_div"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": METHOD,
        "implementation": "equation-faithful independent reproduction",
        "paper": "arXiv:2605.20879v1",
        "official_code_status_at_freeze": "no official code link found in paper",
        "graphs": GRAPHS,
        "mechanisms": MECHANISMS,
        "seeds": SEEDS,
        "anomaly_fraction": ANOMALY_FRACTION,
        "projection_rank": PROJECTION_RANK,
        "svd": "sklearn randomized_svd with fixed random_state=0",
        "neighbor_pairs": "exhaustive unordered pairs",
        "calibration": "absolute deviation from valid-node median, then valid-node z-score",
        "low_degree_score": 0.0,
        "target_labels": "loaded only after scores are constructed; evaluation only",
        "target_specific_tuning": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn_version,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    metrics_path = output / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["seed"] = int(row["seed"])
            for key in ("auroc", "ap", "prevalence", "median_diversity", "seconds"):
                row[key] = float(row[key])
            row["valid_nodes"] = int(row["valid_nodes"])
    else:
        rows = []
    completed_keys = {
        (row["graph"], row["mechanism"], row["seed"]) for row in rows
    }
    verification_path = output / "verification.json"
    verifications = (
        json.loads(verification_path.read_text())
        if verification_path.exists()
        else []
    )
    started = time.perf_counter()
    for graph in GRAPHS:
        raw_features, base_adjacency = load_planetoid(graph, SOURCE_ROOT / "data")
        normalized, _, _ = preprocess(raw_features)
        embedding = projected_features(raw_features)
        for seed in SEEDS:
            for mechanism_id, mechanism in enumerate(MECHANISMS):
                if (graph, mechanism, seed) in completed_keys:
                    continue
                adjacency, anchors, expected_labels, spillover, modification = reconstruct(
                    graph,
                    mechanism,
                    mechanism_id,
                    seed,
                    normalized,
                    base_adjacency,
                )
                scores, diversity, valid, reference = neighbor_div_scores(
                    embedding, adjacency
                )

                # Load labels only after score construction and verify reconstruction.
                labels, frozen_label_source = frozen_labels(
                    graph, mechanism, seed, len(expected_labels)
                )
                saved_anchors = np.flatnonzero(labels)
                np.testing.assert_array_equal(labels, expected_labels)
                np.testing.assert_array_equal(saved_anchors, anchors)

                auroc = float(roc_auc_score(labels, scores))
                ap = float(average_precision_score(labels, scores))
                row = {
                    "method": METHOD,
                    "graph": graph,
                    "mechanism": mechanism,
                    "seed": seed,
                    "auroc": auroc,
                    "ap": ap,
                    "prevalence": float(labels.mean()),
                    "valid_nodes": int(valid.sum()),
                    "median_diversity": float(reference),
                    "seconds": float(time.perf_counter() - started),
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
                    "anomaly_anchors": int(labels.sum()),
                    "valid_nodes": int(valid.sum()),
                    "directly_touched_normal_nodes": len(set(spillover) - set(anchors)),
                    "modification_type": modification.get("selection", mechanism),
                })
                destination = output / graph / mechanism / f"seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination / "scores.npz",
                    scores=scores,
                    diversity=diversity,
                    valid_mask=valid,
                )
                write_csv(metrics_path, rows)
                write_csv(output / "summary.csv", summarize(rows))
                (output / "verification.json").write_text(
                    json.dumps(verifications, indent=2)
                )
                print("RESULT " + json.dumps(row), flush=True)

    summary = summarize(rows)
    write_csv(output / "summary.csv", summary)
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
