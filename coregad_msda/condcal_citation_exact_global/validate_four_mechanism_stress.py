"""Independent validation of the four-mechanism structural stress test."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "four_mechanism_stress_results"
SOURCE_ROOT = ROOT.parent / "condcal_cora"
sys.path.insert(0, str(SOURCE_ROOT))
from condcal import load_planetoid  # noqa: E402

GRAPHS = ["cora", "citeseer", "pubmed"]
MECHANISMS = [
    "cross_community_edges", "degree_preserving_rewire",
    "ego_edge_deletion", "hub_star_injection",
]
SEEDS = [11, 23, 37, 53, 71]
VARIANTS = [
    "contextual_exact_local", "ego_density_evidence",
    "unexpected_edge_evidence", "motif_disruption_evidence",
    "structural_fixed_mean", "structural_without_ego",
    "structural_without_edge", "structural_without_motif",
    "dual_branch_max",
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixed_mean(evidence, mask, columns):
    evidence = evidence[:, columns]
    mask = mask[:, columns]
    counts = mask.sum(axis=1)
    return np.divide(
        (evidence * mask).sum(axis=1), counts,
        out=np.zeros(len(evidence)), where=counts > 0,
    )


def reconstruct_adjacency(base, mechanism, modification):
    adjacency = base.tolil(copy=True)
    if mechanism in {"cross_community_edges", "hub_star_injection"}:
        for first, second in modification["added_edges"]:
            if adjacency[first, second] != 0:
                raise AssertionError("recorded added edge already existed")
            adjacency[first, second] = 1
            adjacency[second, first] = 1
    elif mechanism == "ego_edge_deletion":
        for first, second in modification["removed_edges"]:
            if adjacency[first, second] == 0:
                raise AssertionError("recorded removed edge was absent")
            adjacency[first, second] = 0
            adjacency[second, first] = 0
    elif mechanism == "degree_preserving_rewire":
        for swap in modification["swaps"]:
            for first, second in swap["removed"]:
                if adjacency[first, second] == 0:
                    raise AssertionError("rewire removal was absent")
                adjacency[first, second] = 0
                adjacency[second, first] = 0
            for first, second in swap["added"]:
                if adjacency[first, second] != 0:
                    raise AssertionError("rewire addition already existed")
                adjacency[first, second] = 1
                adjacency[second, first] = 1
    else:
        raise ValueError(mechanism)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    if (adjacency != adjacency.T).nnz:
        raise AssertionError("asymmetric reconstructed graph")
    if adjacency.diagonal().any():
        raise AssertionError("self-loop in reconstructed graph")
    return adjacency


def main():
    metric_rows = read_csv(RESULTS / "metrics.csv")
    metric_index = {
        (int(row["seed"]), row["graph"], row["mechanism"],
         row["population"], row["variant"]): row
        for row in metric_rows
    }
    max_score_error = 0.0
    max_metric_error = 0.0
    rank_checks = 0
    modification_checks = []

    for graph in GRAPHS:
        _, base = load_planetoid(graph, SOURCE_ROOT / "data")
        base_degree = np.diff(base.indptr)
        for seed in SEEDS:
            for mechanism in MECHANISMS:
                directory = RESULTS / graph / mechanism / f"seed_{seed}"
                arrays = np.load(directory / "evidence_and_scores.npz")
                modification = json.loads(
                    (directory / "modification.json").read_text()
                )
                adjacency = reconstruct_adjacency(base, mechanism, modification)
                if mechanism == "degree_preserving_rewire":
                    np.testing.assert_array_equal(
                        np.diff(adjacency.indptr), base_degree
                    )

                labels = arrays["labels"]
                anchors = arrays["anchors"]
                spillover = arrays["spillover"]
                expected_labels = np.zeros(len(labels), dtype=np.int8)
                expected_labels[anchors] = 1
                np.testing.assert_array_equal(labels, expected_labels)
                expected_clean = labels == 0
                expected_clean[spillover] = False
                np.testing.assert_array_equal(
                    arrays["clean_normal_mask"], expected_clean
                )

                local_evidence = np.abs(arrays["local_rank"]) * arrays["local_mask"]
                structural_evidence = (
                    np.abs(arrays["structural_rank"]) * arrays["structural_mask"]
                )
                contextual = fixed_mean(
                    local_evidence, arrays["local_mask"], list(range(5))
                )
                structural = fixed_mean(
                    structural_evidence, arrays["structural_mask"], [0, 1, 2]
                )
                expected_scores = {
                    "contextual_exact_local": contextual,
                    "ego_density_evidence": structural_evidence[:, 0],
                    "unexpected_edge_evidence": structural_evidence[:, 1],
                    "motif_disruption_evidence": structural_evidence[:, 2],
                    "structural_fixed_mean": structural,
                    "structural_without_ego": fixed_mean(
                        structural_evidence, arrays["structural_mask"], [1, 2]
                    ),
                    "structural_without_edge": fixed_mean(
                        structural_evidence, arrays["structural_mask"], [0, 2]
                    ),
                    "structural_without_motif": fixed_mean(
                        structural_evidence, arrays["structural_mask"], [0, 1]
                    ),
                    "dual_branch_max": np.maximum(contextual, structural),
                }
                predictions = read_csv(directory / "predictions.csv")
                for variant, score in expected_scores.items():
                    saved = arrays[f"score_{variant}"]
                    csv_score = np.asarray([
                        float(row[variant]) for row in predictions
                    ])
                    error = max(
                        float(np.max(np.abs(saved - score))),
                        float(np.max(np.abs(csv_score - score))),
                    )
                    max_score_error = max(max_score_error, error)
                    np.testing.assert_allclose(saved, score, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(csv_score, score, rtol=0, atol=1e-15)

                statistics = np.column_stack((
                    arrays["local_statistics"], arrays["structural_statistics"]
                ))
                ranks = np.column_stack((
                    arrays["local_rank"], arrays["structural_rank"]
                ))
                masks = np.column_stack((
                    arrays["local_mask"], arrays["structural_mask"]
                ))
                counts = np.column_stack((
                    arrays["local_counts"], arrays["structural_counts"]
                ))
                regular_nodes = np.linspace(0, len(labels) - 1, 3, dtype=int)
                query_nodes = np.unique(np.concatenate((anchors[:3], regular_nodes)))
                for node in query_nodes:
                    excluded = np.zeros(len(labels), dtype=bool)
                    excluded[node] = True
                    excluded[adjacency.indices[
                        adjacency.indptr[node]:adjacency.indptr[node + 1]
                    ]] = True
                    for statistic_id in range(statistics.shape[1]):
                        value = statistics[node, statistic_id]
                        if not np.isfinite(value):
                            assert masks[node, statistic_id] == 0
                            continue
                        reference = statistics[~excluded, statistic_id]
                        reference = reference[np.isfinite(reference)]
                        expected_rank = 2 * (
                            np.sum(reference < value)
                            + 0.5 * np.sum(reference == value)
                        ) / len(reference) - 1
                        assert counts[node, statistic_id] == len(reference)
                        np.testing.assert_allclose(
                            ranks[node, statistic_id], expected_rank,
                            rtol=0, atol=1e-15,
                        )
                        rank_checks += 1

                populations = {
                    "all_nodes": np.ones(len(labels), dtype=bool),
                    "clean_normal_comparison": (labels == 1) | expected_clean,
                }
                for population, selected in populations.items():
                    y = labels[selected]
                    for variant, score in expected_scores.items():
                        saved = metric_index[
                            (seed, graph, mechanism, population, variant)
                        ]
                        recomputed = {
                            "auroc": roc_auc_score(y, score[selected]),
                            "ap": average_precision_score(y, score[selected]),
                        }
                        for metric, value in recomputed.items():
                            error = abs(float(saved[metric]) - float(value))
                            max_metric_error = max(max_metric_error, error)
                            if error > 1e-14:
                                raise AssertionError("metric mismatch")
                modification_checks.append({
                    "graph": graph, "seed": seed, "mechanism": mechanism,
                    "nodes": len(labels), "anchors": int(labels.sum()),
                    "degree_sequence_preserved": (
                        True if mechanism == "degree_preserving_rewire" else None
                    ),
                    "symmetric_loop_free_graph": True,
                })

    completion = json.loads((RESULTS / "completion.json").read_text())
    assert completion["graph_mechanism_seed_runs"] == completion["expected_runs"] == 60
    assert completion["metric_rows"] == completion["expected_metric_rows"] == 1080
    validation = {
        "status": "passed",
        "graph_mechanism_seed_runs_checked": len(modification_checks),
        "score_vectors_checked": 60 * len(VARIANTS),
        "metric_rows_checked": len(metric_rows),
        "brute_force_exact_rank_checks": rank_checks,
        "maximum_score_absolute_error": max_score_error,
        "maximum_metric_absolute_error": max_metric_error,
        "modification_checks": modification_checks,
    }
    (RESULTS / "independent_validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    print(json.dumps({
        key: value for key, value in validation.items()
        if key != "modification_checks"
    }, indent=2))


if __name__ == "__main__":
    main()
