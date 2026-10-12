"""Four-mechanism structural stress test for exact rank evidence.

The protocol is frozen before reading any new outcome.  It evaluates the same
label-free structural evidence on four graph interventions that do not create
cliques: cross-community edge addition, degree-preserving edge rewiring,
ego-edge deletion, and hub/star injection.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import gc
import json
import platform
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "condcal_cora"
sys.path.insert(0, str(SOURCE_ROOT))
from condcal import (  # noqa: E402
    Config, GRAPH_IDS, STAT_NAMES, extract, load_planetoid, preprocess,
    rng_for, write_csv,
)
from run_exact_global_citation import exact_marginal  # noqa: E402

SEEDS = [11, 23, 37, 53, 71]
GRAPHS = ["cora", "citeseer", "pubmed"]
MECHANISMS = [
    "cross_community_edges",
    "degree_preserving_rewire",
    "ego_edge_deletion",
    "hub_star_injection",
]
STRUCTURAL_NAMES = [
    "ego_triangle_load",
    "unexpected_edge_tail",
    "motif_incoherence",
]
VARIANTS = [
    "contextual_exact_local",
    "ego_density_evidence",
    "unexpected_edge_evidence",
    "motif_disruption_evidence",
    "structural_fixed_mean",
    "structural_without_ego",
    "structural_without_edge",
    "structural_without_motif",
    "dual_branch_max",
]
ANOMALY_FRACTION = 0.05
CROSS_EDGES = 4
HUB_EDGES = 8
REWIRE_SWAPS = 4
EGO_DELETIONS = 4


def candidate_nodes(node, adjacency, forbidden, rng, required, sample_size):
    """Return non-neighbor candidates without allocating a full N-vector."""
    neighbor_set = set(adjacency.rows[node])
    candidates = []
    seen = set()
    attempts = 0
    maximum_attempts = max(2000, 50 * sample_size)
    while len(candidates) < sample_size and attempts < maximum_attempts:
        value = int(rng.integers(adjacency.shape[0]))
        attempts += 1
        if (
            value == node or value in neighbor_set or value in forbidden
            or value in seen
        ):
            continue
        seen.add(value)
        candidates.append(value)
    if len(candidates) < required:
        for value in range(adjacency.shape[0]):
            if (
                value != node and value not in neighbor_set
                and value not in forbidden and value not in seen
            ):
                candidates.append(value)
                if len(candidates) >= required:
                    break
    if len(candidates) < required:
        raise AssertionError("insufficient non-neighbor candidates")
    return np.asarray(candidates, dtype=np.int64)


def select_anchors(n, count, eligible, rng):
    eligible_ids = np.flatnonzero(eligible)
    if len(eligible_ids) < count:
        raise AssertionError("insufficient eligible anomaly anchors")
    return np.sort(rng.choice(eligible_ids, count, replace=False))


def inject_cross_community(base, features, count, rng):
    adjacency = base.tolil(copy=True)
    anchors = select_anchors(len(features), count, np.ones(len(features), bool), rng)
    anchor_set = set(int(value) for value in anchors)
    added = []
    spillover = set()
    for node in anchors:
        for _ in range(CROSS_EDGES):
            candidates = candidate_nodes(
                int(node), adjacency, anchor_set, rng, 1, 64
            )
            dissimilarity = 1.0 - features[candidates] @ features[node]
            other = int(candidates[np.argmax(dissimilarity)])
            adjacency[node, other] = 1
            adjacency[other, node] = 1
            added.append([int(node), other])
            spillover.add(other)
    return adjacency.tocsr(), anchors, spillover, {
        "added_edges": added,
        "edges_per_anchor": CROSS_EDGES,
        "candidate_sample": 64,
        "selection": "maximum cosine dissimilarity among sampled non-neighbors",
    }


def inject_hub_star(base, features, count, rng):
    adjacency = base.tolil(copy=True)
    anchors = select_anchors(len(features), count, np.ones(len(features), bool), rng)
    anchor_set = set(int(value) for value in anchors)
    added = []
    spillover = set()
    for node in anchors:
        candidates = candidate_nodes(
            int(node), adjacency, anchor_set, rng, HUB_EDGES, 4 * HUB_EDGES
        )
        selected = rng.choice(candidates, HUB_EDGES, replace=False)
        for other in selected:
            other = int(other)
            adjacency[node, other] = 1
            adjacency[other, node] = 1
            added.append([int(node), other])
            spillover.add(other)
    return adjacency.tocsr(), anchors, spillover, {
        "added_edges": added,
        "edges_per_anchor": HUB_EDGES,
        "selection": "uniform non-neighbors; no neighbor-neighbor edges added",
    }


def inject_degree_preserving_rewire(base, features, count, rng):
    del features
    adjacency = base.tolil(copy=True)
    base_degree = np.diff(base.indptr)
    anchors = select_anchors(len(base_degree), count, base_degree > 0, rng)
    anchor_set = set(int(value) for value in anchors)
    upper = np.vstack(base.nonzero()).T
    upper = upper[upper[:, 0] < upper[:, 1]]
    spillover = set()
    swaps = []
    per_anchor = {}
    for node_value in anchors:
        node = int(node_value)
        requested = min(REWIRE_SWAPS, int(base_degree[node]))
        completed = 0
        attempts = 0
        while completed < requested and attempts < 2000:
            attempts += 1
            current_neighbors = list(adjacency.rows[node])
            if not current_neighbors:
                break
            normal_neighbors = [u for u in current_neighbors if u not in anchor_set]
            u = int(rng.choice(normal_neighbors or current_neighbors))
            v, w = map(int, upper[int(rng.integers(len(upper)))])
            if v in anchor_set or w in anchor_set:
                continue
            if len({node, u, v, w}) < 4 or adjacency[v, w] == 0:
                continue
            orientations = [(v, w), (w, v)]
            rng.shuffle(orientations)
            chosen = None
            for new_neighbor, paired_neighbor in orientations:
                if adjacency[node, new_neighbor] == 0 and adjacency[u, paired_neighbor] == 0:
                    chosen = (new_neighbor, paired_neighbor)
                    break
            if chosen is None:
                continue
            new_neighbor, paired_neighbor = chosen
            adjacency[node, u] = 0
            adjacency[u, node] = 0
            adjacency[v, w] = 0
            adjacency[w, v] = 0
            adjacency[node, new_neighbor] = 1
            adjacency[new_neighbor, node] = 1
            adjacency[u, paired_neighbor] = 1
            adjacency[paired_neighbor, u] = 1
            swaps.append({
                "anchor": node, "removed": [[node, u], [v, w]],
                "added": [[node, new_neighbor], [u, paired_neighbor]],
            })
            spillover.update([u, v, w, new_neighbor, paired_neighbor])
            completed += 1
        if completed == 0:
            raise AssertionError(f"rewiring failed for anchor {node}")
        per_anchor[str(node)] = completed
    output = adjacency.tocsr()
    output.eliminate_zeros()
    output.sort_indices()
    np.testing.assert_array_equal(np.diff(output.indptr), base_degree)
    return output, anchors, spillover, {
        "swaps": swaps,
        "maximum_swaps_per_anchor": REWIRE_SWAPS,
        "completed_swaps_per_anchor": per_anchor,
        "degree_sequence_preserved_exactly": True,
        "selection": "uniform valid double-edge swaps",
    }


def ego_edges(adjacency, node):
    neighbors = adjacency.rows[node]
    result = []
    for offset, first in enumerate(neighbors):
        first_neighbors = set(adjacency.rows[first])
        for second in neighbors[offset + 1:]:
            if second in first_neighbors:
                result.append((int(first), int(second)))
    return result


def inject_ego_edge_deletion(base, features, count, rng):
    del features
    adjacency = base.tolil(copy=True)
    initial_eligible = np.asarray([
        len(ego_edges(adjacency, node)) > 0 for node in range(base.shape[0])
    ])
    candidates = rng.permutation(np.flatnonzero(initial_eligible))
    anchors = []
    removed = []
    spillover = set()
    per_anchor = {}
    for node_value in candidates:
        if len(anchors) == count:
            break
        node = int(node_value)
        available = ego_edges(adjacency, node)
        if not available:
            continue
        take = min(EGO_DELETIONS, len(available))
        chosen_ids = rng.choice(len(available), take, replace=False)
        chosen = [available[int(index)] for index in chosen_ids]
        for first, second in chosen:
            adjacency[first, second] = 0
            adjacency[second, first] = 0
            removed.append([first, second])
            spillover.update([first, second])
        anchors.append(node)
        per_anchor[str(node)] = take
    if len(anchors) != count:
        raise AssertionError("insufficient ego-edge deletion anchors")
    output = adjacency.tocsr()
    output.eliminate_zeros()
    output.sort_indices()
    anchors = np.sort(np.asarray(anchors, dtype=np.int64))
    return output, anchors, spillover, {
        "removed_edges": removed,
        "maximum_deletions_per_anchor": EGO_DELETIONS,
        "completed_deletions_per_anchor": per_anchor,
        "eligible_nodes_before_injection": int(initial_eligible.sum()),
        "selection": "uniform nodes with at least one existing ego edge",
    }


INJECTORS = {
    "cross_community_edges": inject_cross_community,
    "degree_preserving_rewire": inject_degree_preserving_rewire,
    "ego_edge_deletion": inject_ego_edge_deletion,
    "hub_star_injection": inject_hub_star,
}


def structural_statistics(features, adjacency):
    degree = np.diff(adjacency.indptr)
    result = np.full((len(features), 3), np.nan, dtype=np.float64)
    triangle_counts = np.zeros(len(features), dtype=np.int32)
    for node in range(len(features)):
        neighbors = adjacency.indices[
            adjacency.indptr[node]:adjacency.indptr[node + 1]
        ]
        d = len(neighbors)
        result[node, 0] = 0.0
        if not d:
            continue
        incident = np.clip(1.0 - features[neighbors] @ features[node], 0.0, 2.0)
        tail_size = max(1, int(np.ceil(np.sqrt(d))))
        result[node, 1] = float(np.mean(np.partition(incident, -tail_size)[-tail_size:]))
        if d < 2:
            continue
        ego = adjacency[neighbors][:, neighbors].tocoo()
        keep = ego.row < ego.col
        first = neighbors[ego.row[keep]]
        second = neighbors[ego.col[keep]]
        triangle_counts[node] = len(first)
        result[node, 0] = len(first) / d
        if len(first):
            incoherence = (
                1.0 - features[first] @ features[node]
                + 1.0 - features[second] @ features[node]
                + 1.0 - np.einsum("ij,ij->i", features[first], features[second])
            ) / 3.0
            result[node, 2] = float(np.mean(np.clip(incoherence, 0.0, 2.0)))
    return result, triangle_counts


def fixed_mean(evidence, mask, columns=None):
    if columns is not None:
        evidence = evidence[:, columns]
        mask = mask[:, columns]
    count = mask.sum(axis=1)
    return np.divide(
        (evidence * mask).sum(axis=1), count,
        out=np.zeros(len(evidence), dtype=np.float64), where=count > 0,
    )


def construct_scores(features, adjacency, seed, graph_name, mechanism_id):
    """Construct every score without a label argument."""
    local_statistics, _, _ = extract(
        features, adjacency, Config(),
        rng_for(seed, GRAPH_IDS[graph_name], 200 + mechanism_id),
    )
    local_rank, local_mask, local_counts, local_checks = exact_marginal(
        local_statistics, adjacency
    )
    local_evidence = np.abs(local_rank) * local_mask
    contextual = fixed_mean(local_evidence, local_mask)

    raw_structural, triangle_counts = structural_statistics(features, adjacency)
    structural_rank, structural_mask, structural_counts, structural_checks = exact_marginal(
        raw_structural, adjacency
    )
    structural_evidence = np.abs(structural_rank) * structural_mask
    structural = fixed_mean(structural_evidence, structural_mask)
    scores = {
        "contextual_exact_local": contextual,
        "ego_density_evidence": structural_evidence[:, 0],
        "unexpected_edge_evidence": structural_evidence[:, 1],
        "motif_disruption_evidence": structural_evidence[:, 2],
        "structural_fixed_mean": structural,
        "structural_without_ego": fixed_mean(
            structural_evidence, structural_mask, [1, 2]
        ),
        "structural_without_edge": fixed_mean(
            structural_evidence, structural_mask, [0, 2]
        ),
        "structural_without_motif": fixed_mean(
            structural_evidence, structural_mask, [0, 1]
        ),
        "dual_branch_max": np.maximum(contextual, structural),
    }
    arrays = {
        "local_statistics": local_statistics,
        "local_rank": local_rank, "local_mask": local_mask,
        "local_counts": local_counts,
        "structural_statistics": raw_structural,
        "structural_rank": structural_rank,
        "structural_mask": structural_mask,
        "structural_counts": structural_counts,
        "triangle_counts": triangle_counts,
        **{f"score_{name}": score for name, score in scores.items()},
    }
    audit = {
        "local_exact_pairs": local_checks,
        "structural_exact_pairs": structural_checks,
        "any_local_coverage": float((local_mask.sum(axis=1) > 0).mean()),
        "any_structural_coverage": float((structural_mask.sum(axis=1) > 0).mean()),
        **{
            f"{name}_coverage": float(structural_mask[:, index].mean())
            for index, name in enumerate(STRUCTURAL_NAMES)
        },
    }
    return scores, arrays, audit


def evaluate(scores, labels, clean_normal_mask):
    populations = {
        "all_nodes": np.ones(len(labels), dtype=bool),
        "clean_normal_comparison": (labels == 1) | clean_normal_mask,
    }
    rows = []
    for population, selected in populations.items():
        y = labels[selected]
        for variant in VARIANTS:
            rows.append({
                "population": population, "variant": variant,
                "nodes": int(selected.sum()), "anomalies": int(y.sum()),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, scores[variant][selected])),
                "ap": float(average_precision_score(y, scores[variant][selected])),
            })
    return rows


def summarize(rows):
    output = []
    for graph in GRAPHS:
        for mechanism in MECHANISMS:
            for population in ("all_nodes", "clean_normal_comparison"):
                for variant in VARIANTS:
                    selected = [
                        row for row in rows if row["graph"] == graph
                        and row["mechanism"] == mechanism
                        and row["population"] == population
                        and row["variant"] == variant
                    ]
                    if not selected:
                        continue
                    record = {
                        "graph": graph, "mechanism": mechanism,
                        "population": population, "variant": variant,
                        "seeds": len(selected),
                    }
                    for key in ("nodes", "anomalies", "prevalence", "auroc", "ap"):
                        values = np.asarray([row[key] for row in selected], dtype=float)
                        record[key + "_mean"] = float(values.mean())
                        record[key + "_sd"] = (
                            float(values.std(ddof=1)) if len(values) > 1 else 0.0
                        )
                    record["auroc_above_random_seeds"] = int(
                        sum(row["auroc"] > 0.5 for row in selected)
                    )
                    record["ap_above_prevalence_seeds"] = int(
                        sum(row["ap"] > row["prevalence"] for row in selected)
                    )
                    output.append(record)
    return output


def acceptance(summary_rows):
    primary = [row for row in summary_rows if row["population"] == "all_nodes"]
    outcomes = []
    for graph in GRAPHS:
        improvements = 0
        mechanisms_passing = 0
        for mechanism in MECHANISMS:
            rows = {
                row["variant"]: row for row in primary
                if row["graph"] == graph and row["mechanism"] == mechanism
            }
            structural = rows["structural_fixed_mean"]
            contextual = rows["contextual_exact_local"]
            pass_random = (
                structural["auroc_mean"] > 0.5
                and structural["ap_mean"] > structural["prevalence_mean"]
            )
            improves = (
                structural["auroc_mean"] > contextual["auroc_mean"]
                and structural["ap_mean"] > contextual["ap_mean"]
            )
            mechanisms_passing += pass_random
            improvements += improves
            outcomes.append({
                "graph": graph, "mechanism": mechanism,
                "structural_above_random_and_prevalence": bool(pass_random),
                "structural_improves_both_metrics_over_contextual": bool(improves),
                "all_three_leave_one_out_aurocs_above_random": bool(all(
                    rows[variant]["auroc_mean"] > 0.5 for variant in (
                        "structural_without_ego", "structural_without_edge",
                        "structural_without_motif",
                    )
                )),
            })
        outcomes.append({
            "graph": graph,
            "mechanisms_above_random_and_prevalence": mechanisms_passing,
            "mechanisms_improving_both_metrics": improvements,
            "passes_declared_three_of_four_improvement_rule": improvements >= 3,
        })
    return outcomes


def main():
    output = ROOT / "four_mechanism_stress_results"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "frozen_before_results",
        "graphs": GRAPHS, "mechanisms": MECHANISMS, "seeds": SEEDS,
        "anomaly_fraction": ANOMALY_FRACTION,
        "mechanism_parameters": {
            "cross_community_edges_per_anchor": CROSS_EDGES,
            "hub_star_edges_per_anchor": HUB_EDGES,
            "maximum_degree_preserving_swaps_per_anchor": REWIRE_SWAPS,
            "maximum_ego_edges_deleted_per_anchor": EGO_DELETIONS,
        },
        "structural_statistics": STRUCTURAL_NAMES,
        "contextual_statistics": STAT_NAMES,
        "calibration": "absolute exact empirical mid-rank using all valid nodes except query and immediate neighbors",
        "aggregation": "coverage-safe fixed mean; fixed max for diagnostic dual-branch fusion",
        "evaluation_populations": {
            "all_nodes": "primary; includes normal nodes incidentally touched by interventions",
            "clean_normal_comparison": "sensitivity; anomaly anchors versus normal nodes whose adjacency was not directly modified",
        },
        "acceptance": {
            "per_mechanism": "mean AUROC > 0.5 and mean AP > prevalence",
            "improvement": "structural fixed mean improves both AUROC and AP over contextual exact local in at least 3 of 4 mechanisms per graph",
            "ablation": "all three leave-one-coordinate-out mean AUROCs remain above 0.5",
        },
        "target_labels": "created from intervention anchors; passed only to evaluation after all scores are constructed",
        "target_specific_thresholds_or_fitting": False,
        "development_status": "all three citation graphs are development graphs; no definitive real-world claim",
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    bases = {}
    for graph in GRAPHS:
        raw_features, adjacency = load_planetoid(graph, SOURCE_ROOT / "data")
        normalized, _, retained = preprocess(raw_features)
        bases[graph] = (normalized, adjacency, int(retained.sum()))

    rows, audits = [], []
    started = time.perf_counter()
    for graph in GRAPHS:
        features, base_adjacency, retained = bases[graph]
        anomaly_count = int(np.floor(ANOMALY_FRACTION * len(features)))
        for seed in SEEDS:
            for mechanism_id, mechanism in enumerate(MECHANISMS):
                rng = rng_for(seed, GRAPH_IDS[graph], 100 + mechanism_id)
                adjacency, anchors, spillover, modification = INJECTORS[mechanism](
                    base_adjacency, features, anomaly_count, rng
                )
                adjacency.eliminate_zeros()
                adjacency.sort_indices()
                labels = np.zeros(len(features), dtype=np.int8)
                labels[anchors] = 1
                spillover = set(spillover) - set(int(value) for value in anchors)
                clean_normal = labels == 0
                if spillover:
                    clean_normal[np.fromiter(spillover, dtype=np.int64)] = False

                scores, arrays, score_audit = construct_scores(
                    features, adjacency, seed, graph, mechanism_id
                )
                # Synthetic anchor labels are accessed only after all scores exist.
                evaluated = evaluate(scores, labels, clean_normal)
                for row in evaluated:
                    row.update(seed=seed, graph=graph, mechanism=mechanism)
                    rows.append(row)

                destination = output / graph / mechanism / f"seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination / "evidence_and_scores.npz",
                    anchors=anchors, spillover=np.asarray(sorted(spillover), dtype=np.int64),
                    labels=labels, clean_normal_mask=clean_normal,
                    **arrays,
                )
                write_csv(destination / "predictions.csv", [
                    {
                        "node": node, "anomaly_anchor": int(labels[node]),
                        "clean_normal": int(clean_normal[node]),
                        "directly_touched_normal": int(
                            labels[node] == 0 and not clean_normal[node]
                        ),
                        **{variant: float(score[node]) for variant, score in scores.items()},
                    }
                    for node in range(len(labels))
                ])
                (destination / "modification.json").write_text(
                    json.dumps(modification, indent=2)
                )
                audits.append({
                    "graph": graph, "seed": seed, "mechanism": mechanism,
                    "nodes": len(labels), "features_retained": retained,
                    "anomaly_anchors": int(labels.sum()),
                    "directly_touched_normal_nodes": len(spillover),
                    "scores_constructed_before_label_evaluation": True,
                    "target_labels_used_for_evaluation_only": True,
                    **score_audit,
                })
                write_csv(output / "metrics.csv", rows)
                summary_rows = summarize(rows)
                write_csv(output / "summary.csv", summary_rows)
                (output / "verification.json").write_text(json.dumps(audits, indent=2))
                current = [
                    row for row in rows if row["graph"] == graph
                    and row["seed"] == seed and row["mechanism"] == mechanism
                    and row["population"] == "all_nodes"
                    and row["variant"] in (
                        "contextual_exact_local", "structural_fixed_mean",
                        "dual_branch_max",
                    )
                ]
                print("RESULT " + json.dumps(current), flush=True)
                del adjacency, labels, scores, arrays
                gc.collect()

    summary_rows = summarize(rows)
    write_csv(output / "summary.csv", summary_rows)
    (output / "acceptance.json").write_text(
        json.dumps(acceptance(summary_rows), indent=2)
    )
    completion = {
        "status": "complete", "graph_mechanism_seed_runs": len(audits),
        "expected_runs": len(GRAPHS) * len(MECHANISMS) * len(SEEDS),
        "metric_rows": len(rows),
        "expected_metric_rows": len(GRAPHS) * len(MECHANISMS) * len(SEEDS) * 2 * len(VARIANTS),
        "seconds": time.perf_counter() - started,
    }
    (output / "completion.json").write_text(json.dumps(completion, indent=2))
    print("COMPLETE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
