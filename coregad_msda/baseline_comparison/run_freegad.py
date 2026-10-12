"""Leakage-audited FreeGAD reproduction on the frozen citation stress test.

The score follows the official implementation at commit
c1b007d69174e22fcf9d7ff7fea0dcb88ebdf2a0.  The official Cora preset is
transferred unchanged to all three citation graphs because the repository does
not provide CiteSeer or PubMed presets.  Labels are loaded only after scoring.
"""

import csv
import gc
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
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

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

METHOD = "FreeGAD"
OFFICIAL_COMMIT = "c1b007d69174e22fcf9d7ff7fea0dcb88ebdf2a0"
ALPHA = 0.010974988
BETA = 0.205651231
NUM_HOPS = 7
NUM_ANCHORS = 70
DISTANCE_BATCH = 512


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_normalize_features(features):
    features = np.asarray(features, dtype=np.float32)
    sums = features.sum(axis=1, keepdims=True)
    return np.divide(
        features, sums, out=np.zeros_like(features), where=sums != 0
    )


def symmetric_normalize_with_self_loops(adjacency):
    matrix = adjacency.astype(np.float32) + sparse.eye(
        adjacency.shape[0], dtype=np.float32, format="csr"
    )
    matrix = matrix.maximum(matrix.T).tocsr()
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive] ** -0.5
    diagonal = sparse.diags(inv_sqrt, format="csr")
    output = (diagonal @ matrix @ diagonal).tocsr()
    output.sort_indices()
    return output


def affinity_gated_embedding(features, adjacency):
    """Official propagation and affinity-gated residual encoder equations."""
    normalized_adjacency = symmetric_normalize_with_self_loops(adjacency)
    propagated = []
    current = features
    base_norm = np.linalg.norm(features, axis=1) + 1e-10
    cosine_columns = []
    for _ in range(NUM_HOPS):
        current = np.asarray(normalized_adjacency @ current, dtype=np.float32)
        propagated.append(current)
        current_norm = np.linalg.norm(current, axis=1) + 1e-10
        cosine_columns.append(
            np.einsum("ij,ij->i", current, features) / (current_norm * base_norm)
        )
    logits = np.column_stack(cosine_columns).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)

    embedding = np.zeros_like(features, dtype=np.float32)
    for hop, propagated_features in enumerate(propagated):
        weight = weights[:, hop : hop + 1]
        embedding += (1.0 - weight) * propagated_features + weight * features
    embedding /= NUM_HOPS
    return embedding


def distance_summary(query, anchors):
    """Return mean, min and max Euclidean distances in bounded memory."""
    result_mean = np.empty(len(query), dtype=np.float32)
    result_min = np.empty(len(query), dtype=np.float32)
    result_max = np.empty(len(query), dtype=np.float32)
    anchor_norm = np.einsum("ij,ij->i", anchors, anchors)
    for start in range(0, len(query), DISTANCE_BATCH):
        stop = min(start + DISTANCE_BATCH, len(query))
        batch = query[start:stop]
        squared = (
            np.einsum("ij,ij->i", batch, batch)[:, None]
            + anchor_norm[None, :]
            - 2.0 * (batch @ anchors.T)
        )
        np.maximum(squared, 0.0, out=squared)
        distances = np.sqrt(squared, out=squared)
        result_mean[start:stop] = distances.mean(axis=1)
        result_min[start:stop] = distances.min(axis=1)
        result_max[start:stop] = distances.max(axis=1)
    return result_mean, result_min, result_max


def freegad_scores(features, adjacency):
    embedding = affinity_gated_embedding(features, adjacency)
    norms = np.linalg.norm(embedding, axis=1) * np.linalg.norm(features, axis=1)
    similarity = np.divide(
        np.einsum("ij,ij->i", embedding, features),
        norms,
        out=np.zeros(len(features), dtype=np.float32),
        where=norms != 0,
    )
    # torch.topk is unspecified on exact ties. Stable lexicographic ordering is
    # used here and recorded because sparse citation features create ties.
    node_ids = np.arange(len(features))
    positive_ids = np.lexsort((node_ids, -similarity))[:NUM_ANCHORS]
    negative_ids = np.lexsort((node_ids, similarity))[:NUM_ANCHORS]
    pos_mean, pos_min, pos_max = distance_summary(
        embedding, embedding[positive_ids]
    )
    neg_mean, neg_min, neg_max = distance_summary(
        embedding, embedding[negative_ids]
    )
    score = ALPHA * (pos_mean + pos_min + pos_max)
    score -= BETA * (neg_mean + neg_max + neg_min)
    return score.astype(np.float64), similarity, positive_ids, negative_ids


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
    output = ROOT / "results" / "freegad"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": METHOD,
        "implementation": "equation-faithful NumPy/SciPy reproduction of official code",
        "official_commit": OFFICIAL_COMMIT,
        "graphs": GRAPHS,
        "mechanisms": MECHANISMS,
        "seeds": SEEDS,
        "anomaly_fraction": ANOMALY_FRACTION,
        "configuration_source": "official Cora preset transferred unchanged to all citation graphs",
        "alpha": ALPHA,
        "beta": BETA,
        "num_hops": NUM_HOPS,
        "num_anchors_per_pseudo_class": NUM_ANCHORS,
        "features": "row-sum normalized for all citation graphs",
        "topk_ties": "stable similarity ordering with node id as tie-breaker",
        "target_labels": "loaded only after scores are constructed; evaluation only",
        "target_specific_tuning": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows = []
    verifications = []
    started = time.perf_counter()
    for graph in GRAPHS:
        raw_features, base_adjacency = load_planetoid(graph, SOURCE_ROOT / "data")
        normalized, _, _ = preprocess(raw_features)
        features = row_normalize_features(raw_features)
        for seed in SEEDS:
            for mechanism_id, mechanism in enumerate(MECHANISMS):
                adjacency, anchors, expected_labels, spillover, modification = reconstruct(
                    graph,
                    mechanism,
                    mechanism_id,
                    seed,
                    normalized,
                    base_adjacency,
                )
                scores, similarity, positive_ids, negative_ids = freegad_scores(
                    features, adjacency
                )

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
                    "positive_proxy_anchor_overlap_with_true_anomalies": int(
                        labels[positive_ids].sum()
                    ),
                    "negative_proxy_anchor_overlap_with_true_anomalies": int(
                        labels[negative_ids].sum()
                    ),
                    "nodes": len(labels),
                    "edges": int(adjacency.nnz // 2),
                    "directly_touched_normal_nodes": len(set(spillover) - set(anchors)),
                    "modification_type": modification.get("selection", mechanism),
                })
                destination = output / graph / mechanism / f"seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination / "scores.npz",
                    scores=scores,
                    affinity=similarity,
                    positive_proxy_anchors=positive_ids,
                    negative_proxy_anchors=negative_ids,
                )
                write_csv(output / "metrics.csv", rows)
                write_csv(output / "summary.csv", summarize(rows))
                (output / "verification.json").write_text(
                    json.dumps(verifications, indent=2)
                )
                print("RESULT " + json.dumps(row), flush=True)
                del adjacency, scores, similarity
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
