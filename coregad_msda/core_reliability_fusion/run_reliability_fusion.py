"""Training-free reliability-weighted aggregation for CoRe-GAD.

Development and evaluation are deliberately separated.  Candidate aggregation
rules are compared on Cora/CiteSeer seeds 11, 23, and 37.  The selected rule is
written to freeze.json.  Evaluation then reads that file and evaluates only the
frozen rule (plus the previously declared fixed-mean reference) on held-out
citation runs and the Reddit and Amazon Fraud graphs retained in the final protocol.
"""

import argparse
import csv
import gc
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CITATION_ROOT = WORKSPACE / "condcal_citation_exact_global"
CITATION_RESULTS = CITATION_ROOT / "four_mechanism_stress_results"
SOURCE_ROOT = WORKSPACE / "condcal_cora"
AMAZON_ROOT = WORKSPACE / "condcal_amazon"
sys.path.insert(0, str(CITATION_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(AMAZON_ROOT))

from condcal import Config, GRAPH_IDS, extract, load_planetoid, preprocess, rng_for  # noqa: E402
from run_exact_global_citation import exact_marginal  # noqa: E402
from run_four_mechanism_stress import (  # noqa: E402
    ANOMALY_FRACTION,
    INJECTORS,
    MECHANISMS,
    structural_statistics,
)
from dataset_loaders import load_amazon, load_reddit  # noqa: E402

LOCAL_NAMES = [
    "node_neighbor_discrepancy",
    "neighbor_pair_mean",
    "neighbor_pair_variance",
    "closure",
    "neighbor_degree",
]
STRUCTURAL_NAMES = [
    "ego_triangle_load",
    "unexpected_edge_tail",
    "motif_incoherence",
]
COORDINATE_NAMES = LOCAL_NAMES + STRUCTURAL_NAMES
DEV_GRAPHS = ["cora", "citeseer"]
DEV_SEEDS = [11, 23, 37]
INTERNAL_HOLDOUT_SEEDS = [53, 71]
PUBMED_SEEDS = [11, 23, 37, 53, 71]
REAL_SEEDS = [11, 23, 37, 53, 71]
REAL_GRAPH_IDS = {"reddit": 4, "amazon": 6}
REFERENCE_IQR = 0.5  # IQR of Uniform(0,1) absolute rank evidence.
EPSILON = 1e-6
CANDIDATES = [
    "fixed_all_mean",
    "fixed_branch_max",
    "reliability_mean",
    "weighted_stouffer",
    "weighted_cauchy",
    "reliability_max",
    "reliability_branch_max",
    "branch_cauchy_max",
]
REFERENCE_VARIANTS = [
    "fixed_all_mean",
    "fixed_branch_max",
    "fixed_structural_mean",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def masked_mean(values, weights):
    denominator = weights.sum(axis=1)
    return np.divide(
        (values * weights).sum(axis=1),
        denominator,
        out=np.zeros(len(values), dtype=np.float64),
        where=denominator > 0,
    )


def full_signed_midranks(statistics):
    output = np.zeros_like(statistics, dtype=np.float64)
    for coordinate in range(statistics.shape[1]):
        values = statistics[:, coordinate]
        valid = np.isfinite(values)
        count = int(valid.sum())
        if count:
            output[valid, coordinate] = (
                2.0 * (rankdata(values[valid], method="average") - 0.5) / count - 1.0
            )
    return output


def reliability_components(statistics, ranks, mask, counts):
    """Compute label-free node-coordinate reliability.

    Reliability is the product of (i) retained reference fraction, (ii) robust
    evidence dispersion relative to an ideal Uniform(0,1) rank coordinate, and
    (iii) local-reference stability measured against the corresponding full
    marginal midrank.  Every factor is in [0,1].
    """
    evidence = np.abs(ranks) * mask
    valid_counts = mask.sum(axis=0)
    denominator = np.maximum(valid_counts - 1.0, 1.0)
    support = np.clip(counts / denominator[None, :], 0.0, 1.0) * mask

    dispersion = np.zeros(statistics.shape[1], dtype=np.float64)
    for coordinate in range(statistics.shape[1]):
        valid = mask[:, coordinate] > 0
        if valid.any():
            lower, upper = np.quantile(evidence[valid, coordinate], [0.25, 0.75])
            dispersion[coordinate] = np.clip(
                (upper - lower) / REFERENCE_IQR, 0.0, 1.0
            )

    full_ranks = full_signed_midranks(statistics)
    instability = np.abs(ranks - full_ranks) / 2.0
    stability = np.clip(1.0 - instability, 0.0, 1.0) * mask
    weights = support * dispersion[None, :] * stability
    return {
        "evidence": evidence,
        "support": support,
        "dispersion": dispersion,
        "stability": stability,
        "instability": instability * mask,
        "weights": weights,
        "full_ranks": full_ranks,
    }


def candidate_scores(components, mask):
    evidence = components["evidence"]
    weights = components["weights"]
    fixed_local = masked_mean(evidence[:, :5], mask[:, :5])
    fixed_structural = masked_mean(evidence[:, 5:], mask[:, 5:])
    reliability_local = masked_mean(evidence[:, :5], weights[:, :5])
    reliability_structural = masked_mean(evidence[:, 5:], weights[:, 5:])

    clipped = np.clip(evidence, EPSILON, 1.0 - EPSILON)
    z_values = ndtri(clipped)
    stouffer_denominator = np.sqrt(np.square(weights).sum(axis=1))
    stouffer = np.divide(
        (weights * z_values).sum(axis=1),
        stouffer_denominator,
        out=np.zeros(len(evidence), dtype=np.float64),
        where=stouffer_denominator > 0,
    )
    tangent = np.tan(np.pi * (clipped - 0.5))
    cauchy = 0.5 + np.arctan(masked_mean(tangent, weights)) / np.pi
    local_cauchy = 0.5 + np.arctan(
        masked_mean(tangent[:, :5], weights[:, :5])
    ) / np.pi
    structural_cauchy = 0.5 + np.arctan(
        masked_mean(tangent[:, 5:], weights[:, 5:])
    ) / np.pi

    return {
        "fixed_all_mean": masked_mean(evidence, mask),
        "fixed_branch_max": np.maximum(fixed_local, fixed_structural),
        "reliability_mean": masked_mean(evidence, weights),
        "weighted_stouffer": ndtr(stouffer),
        "weighted_cauchy": cauchy,
        "reliability_max": np.max(weights * evidence, axis=1),
        "reliability_branch_max": np.maximum(
            reliability_local, reliability_structural
        ),
        "branch_cauchy_max": np.maximum(local_cauchy, structural_cauchy),
        "fixed_structural_mean": fixed_structural,
        "fixed_contextual_mean": fixed_local,
    }


def citation_arrays(graph, mechanism, seed):
    path = CITATION_RESULTS / graph / mechanism / f"seed_{seed}" / "evidence_and_scores.npz"
    try:
        with np.load(path) as handle:
            arrays = {key: handle[key] for key in (
                "local_statistics", "local_rank", "local_mask", "local_counts",
                "structural_statistics", "structural_rank", "structural_mask",
                "structural_counts",
            )}
        return arrays, "frozen_evidence_archive"
    except Exception:
        raw_features, base_adjacency = load_planetoid(graph, SOURCE_ROOT / "data")
        normalized, _, _ = preprocess(raw_features)
        anomaly_count = int(np.floor(ANOMALY_FRACTION * len(normalized)))
        mechanism_id = MECHANISMS.index(mechanism)
        adjacency, _, _, _ = INJECTORS[mechanism](
            base_adjacency,
            normalized,
            anomaly_count,
            rng_for(seed, GRAPH_IDS[graph], 100 + mechanism_id),
        )
        adjacency.eliminate_zeros()
        adjacency.sort_indices()
        local_statistics, _, _ = extract(
            normalized,
            adjacency,
            Config(),
            rng_for(seed, GRAPH_IDS[graph], 200 + mechanism_id),
        )
        local_rank, local_mask, local_counts, _ = exact_marginal(
            local_statistics, adjacency
        )
        structural_stats, _ = structural_statistics(normalized, adjacency)
        structural_rank, structural_mask, structural_counts, _ = exact_marginal(
            structural_stats, adjacency
        )
        return {
            "local_statistics": local_statistics,
            "local_rank": local_rank,
            "local_mask": local_mask,
            "local_counts": local_counts,
            "structural_statistics": structural_stats,
            "structural_rank": structural_rank,
            "structural_mask": structural_mask,
            "structural_counts": structural_counts,
        }, "deterministically_reconstructed"


def combine_arrays(arrays):
    statistics = np.concatenate(
        [arrays["local_statistics"], arrays["structural_statistics"]], axis=1
    )
    ranks = np.concatenate(
        [arrays["local_rank"], arrays["structural_rank"]], axis=1
    )
    mask = np.concatenate(
        [arrays["local_mask"], arrays["structural_mask"]], axis=1
    )
    counts = np.concatenate(
        [arrays["local_counts"], arrays["structural_counts"]], axis=1
    )
    return statistics, ranks, mask, counts


def citation_labels(graph, mechanism, seed):
    path = CITATION_RESULTS / graph / mechanism / f"seed_{seed}" / "predictions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_nodes = {"cora": 2708, "citeseer": 3327, "pubmed": 19717}[graph]
    if len(rows) == expected_nodes:
        return np.asarray([int(row["anomaly_anchor"]) for row in rows], dtype=np.int8)
    manifest = json.loads((path.parent / "modification.json").read_text())
    if mechanism in ("cross_community_edges", "hub_star_injection"):
        anchors = sorted({int(edge[0]) for edge in manifest["added_edges"]})
    elif mechanism == "degree_preserving_rewire":
        anchors = sorted(int(key) for key in manifest["completed_swaps_per_anchor"])
    else:
        anchors = sorted(int(key) for key in manifest["completed_deletions_per_anchor"])
    labels = np.zeros(expected_nodes, dtype=np.int8)
    labels[anchors] = 1
    return labels


def build_real_arrays(name, raw_features, adjacency, seed):
    normalized, _, _ = preprocess(raw_features)
    local_statistics, _, _ = extract(
        normalized,
        adjacency,
        Config(),
        np.random.default_rng(np.random.SeedSequence([seed, REAL_GRAPH_IDS[name], 200])),
    )
    local_rank, local_mask, local_counts, _ = exact_marginal(
        local_statistics, adjacency
    )
    structural_stats, _ = structural_statistics(normalized, adjacency)
    structural_rank, structural_mask, structural_counts, _ = exact_marginal(
        structural_stats, adjacency
    )
    return {
        "local_statistics": local_statistics,
        "local_rank": local_rank,
        "local_mask": local_mask,
        "local_counts": local_counts,
        "structural_statistics": structural_stats,
        "structural_rank": structural_rank,
        "structural_mask": structural_mask,
        "structural_counts": structural_counts,
    }


def score_arrays(arrays):
    statistics, ranks, mask, counts = combine_arrays(arrays)
    components = reliability_components(statistics, ranks, mask, counts)
    scores = candidate_scores(components, mask)
    return scores, components, mask


def metric_row(dataset, mechanism, seed, variant, labels, scores, phase):
    return {
        "phase": phase,
        "dataset": dataset,
        "mechanism": mechanism,
        "seed": seed,
        "variant": variant,
        "nodes": len(labels),
        "anomalies": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def summary_rows(rows):
    output = []
    groups = sorted({(r["phase"], r["dataset"], r["variant"]) for r in rows})
    for phase, dataset, variant in groups:
        selected = [
            r for r in rows
            if r["phase"] == phase and r["dataset"] == dataset
            and r["variant"] == variant
        ]
        output.append({
            "phase": phase,
            "dataset": dataset,
            "variant": variant,
            "runs": len(selected),
            "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
            "auroc_std": float(np.std([r["auroc"] for r in selected])),
            "ap_mean": float(np.mean([r["ap"] for r in selected])),
            "ap_std": float(np.std([r["ap"] for r in selected])),
            "prevalence_mean": float(np.mean([r["prevalence"] for r in selected])),
        })
    return output


def reliability_audit(dataset, mechanism, seed, components, mask, source):
    rows = []
    for coordinate, name in enumerate(COORDINATE_NAMES):
        valid = mask[:, coordinate] > 0
        rows.append({
            "dataset": dataset,
            "mechanism": mechanism,
            "seed": seed,
            "coordinate": name,
            "evidence_source": source,
            "coverage": float(valid.mean()),
            "dispersion_reliability": float(components["dispersion"][coordinate]),
            "mean_support_reliability": float(
                components["support"][valid, coordinate].mean()
            ) if valid.any() else 0.0,
            "mean_stability_reliability": float(
                components["stability"][valid, coordinate].mean()
            ) if valid.any() else 0.0,
            "mean_combined_reliability": float(
                components["weights"][valid, coordinate].mean()
            ) if valid.any() else 0.0,
        })
    return rows


def run_development(output):
    protocol = {
        "status": "exploratory aggregation development; frozen before evaluation scoring",
        "development_graphs": DEV_GRAPHS,
        "development_seeds": DEV_SEEDS,
        "development_mechanisms": MECHANISMS,
        "candidate_rules": CANDIDATES,
        "selection_criterion": "maximize mean of normalized AUROC gain above 0.5 and normalized AP gain above run prevalence",
        "selection_formula": "mean_run(0.5*((AUROC-0.5)/0.5 + (AP-prevalence)/(1-prevalence)))",
        "tie_breaking": "higher mean AUROC, then higher mean AP, then lexicographic name",
        "evaluation_not_accessed_during_selection": True,
        "reliability": {
            "support": "exact retained reference fraction n_ik/(N_valid,k-1)",
            "dispersion": "min(1, IQR(|rank|)/0.5)",
            "stability": "1-|local neighborhood-excluded rank - full marginal rank|/2",
            "combined": "product of support, dispersion, and stability",
        },
    }
    (output / "development_protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    audits = []
    for graph in DEV_GRAPHS:
        for mechanism in MECHANISMS:
            for seed in DEV_SEEDS:
                arrays, source = citation_arrays(graph, mechanism, seed)
                scores, components, mask = score_arrays(arrays)
                # Labels are accessed only after every candidate score exists.
                labels = citation_labels(graph, mechanism, seed)
                for variant in CANDIDATES:
                    rows.append(metric_row(
                        graph, mechanism, seed, variant, labels,
                        scores[variant], "development",
                    ))
                audits.extend(reliability_audit(
                    graph, mechanism, seed, components, mask, source
                ))
    write_csv(output / "development_metrics.csv", rows)
    write_csv(output / "development_reliability.csv", audits)

    candidate_summary = []
    for variant in CANDIDATES:
        selected = [row for row in rows if row["variant"] == variant]
        utility = np.mean([
            0.5 * (
                (row["auroc"] - 0.5) / 0.5
                + (row["ap"] - row["prevalence"]) / (1.0 - row["prevalence"])
            )
            for row in selected
        ])
        candidate_summary.append({
            "variant": variant,
            "runs": len(selected),
            "selection_utility": float(utility),
            "auroc_mean": float(np.mean([row["auroc"] for row in selected])),
            "ap_mean": float(np.mean([row["ap"] for row in selected])),
        })
    candidate_summary.sort(
        key=lambda row: (
            -row["selection_utility"], -row["auroc_mean"],
            -row["ap_mean"], row["variant"],
        )
    )
    write_csv(output / "development_candidate_summary.csv", candidate_summary)
    selected = candidate_summary[0]["variant"]
    freeze = {
        "status": "frozen_before_evaluation_phase",
        "selected_variant": selected,
        "selected_development_utility": candidate_summary[0]["selection_utility"],
        "selected_development_auroc": candidate_summary[0]["auroc_mean"],
        "selected_development_ap": candidate_summary[0]["ap_mean"],
        "candidate_order": [row["variant"] for row in candidate_summary],
        "evaluation_variants": [selected, *REFERENCE_VARIANTS],
        "evaluation_sets": {
            "internal_holdout": "Cora and CiteSeer seeds 53 and 71, four mechanisms",
            "citation_transfer": "PubMed seeds 11,23,37,53,71, four mechanisms",
            "organic_labels": "Reddit and Amazon Fraud; five statistic-sampling seeds",
        },
    }
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2))
    return selected


def evaluate_citation(output, selected_variant, rows, audits):
    specifications = [
        (graph, seed)
        for graph in DEV_GRAPHS for seed in INTERNAL_HOLDOUT_SEEDS
    ] + [("pubmed", seed) for seed in PUBMED_SEEDS]
    for graph, seed in specifications:
        phase = "internal_holdout" if graph in DEV_GRAPHS else "citation_transfer"
        for mechanism in MECHANISMS:
            started = time.perf_counter()
            arrays, source = citation_arrays(graph, mechanism, seed)
            scores, components, mask = score_arrays(arrays)
            aggregation_seconds = time.perf_counter() - started
            labels = citation_labels(graph, mechanism, seed)
            for variant in (selected_variant, *REFERENCE_VARIANTS):
                row = metric_row(
                    graph, mechanism, seed, variant, labels, scores[variant], phase
                )
                row["construction_seconds"] = 0.0
                row["aggregation_seconds"] = aggregation_seconds
                rows.append(row)
            audits.extend(reliability_audit(
                graph, mechanism, seed, components, mask, source
            ))


def evaluate_real(output, selected_variant, rows, audits):
    loaders = {"reddit": load_reddit, "amazon": load_amazon}
    data_root = AMAZON_ROOT / "data"
    for name, loader in loaders.items():
        loader_root = data_root
        raw_features, adjacency, labels, labeled = loader(loader_root)
        dataset_output = output / name
        dataset_output.mkdir(parents=True, exist_ok=True)
        for seed in REAL_SEEDS:
            started = time.perf_counter()
            arrays = build_real_arrays(name, raw_features, adjacency, seed)
            construction_seconds = time.perf_counter() - started
            aggregation_started = time.perf_counter()
            scores, components, mask = score_arrays(arrays)
            aggregation_seconds = time.perf_counter() - aggregation_started
            selected_nodes = labeled if name == "amazon" else np.ones(len(labels), bool)
            evaluation_labels = labels[selected_nodes]
            for variant in (selected_variant, *REFERENCE_VARIANTS):
                row = metric_row(
                    name, "organic_labels", seed, variant,
                    evaluation_labels, scores[variant][selected_nodes], "organic_labels",
                )
                row["construction_seconds"] = construction_seconds
                row["aggregation_seconds"] = aggregation_seconds
                rows.append(row)
            audits.extend(reliability_audit(
                name, "organic_labels", seed, components, mask, "fresh_exact_calibration"
            ))
            np.savez_compressed(
                dataset_output / f"seed_{seed}_scores.npz",
                selected_score=scores[selected_variant],
                fixed_structural_mean=scores["fixed_structural_mean"],
                reliability_weights=components["weights"],
                reliability_dispersion=components["dispersion"],
                mask=mask,
            )
            gc.collect()


def run_evaluation(output):
    freeze_path = output / "freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("development must complete and freeze.json must exist")
    freeze = json.loads(freeze_path.read_text())
    if freeze["status"] != "frozen_before_evaluation_phase":
        raise RuntimeError("aggregation rule is not frozen")
    selected_variant = freeze["selected_variant"]
    started = time.perf_counter()
    rows, audits = [], []
    evaluate_citation(output, selected_variant, rows, audits)
    evaluate_real(output, selected_variant, rows, audits)
    write_csv(output / "evaluation_metrics.csv", rows)
    write_csv(output / "evaluation_summary.csv", summary_rows(rows))
    write_csv(output / "evaluation_reliability.csv", audits)
    completion = {
        "status": "complete",
        "selected_variant": selected_variant,
        "evaluation_metric_rows": len(rows),
        "citation_run_variant_rows": (1 + len(REFERENCE_VARIANTS)) * (16 + 20),
        "organic_run_variant_rows": (1 + len(REFERENCE_VARIANTS)) * 15,
        "target_labels_used_for_aggregation": False,
        "trainable_parameters": 0,
        "training_required": False,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn_version,
    }
    (output / "evaluation_completion.json").write_text(
        json.dumps(completion, indent=2)
    )
    return selected_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("development", "evaluation", "all"), default="all", nargs="?"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("development", "all"):
        selected = run_development(args.output)
        print("FROZEN " + selected, flush=True)
    if args.phase in ("evaluation", "all"):
        selected = run_evaluation(args.output)
        print("COMPLETE " + selected, flush=True)


if __name__ == "__main__":
    main()
