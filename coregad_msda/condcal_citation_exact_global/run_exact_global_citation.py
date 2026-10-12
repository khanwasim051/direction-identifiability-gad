"""Leakage-audited exact local/global diagnostic on Cora and CiteSeer.

This is a frozen, label-free scoring experiment.  For each seed, it reproduces
the previously declared synthetic anomaly injection, reuses the corresponding
five local statistics, and compares three scores:

1. exact_local: mean absolute exact empirical mid-rank over valid local stats;
2. exact_global: absolute exact empirical mid-rank of global feature radius;
3. exact_local_global: coverage-safe mean of all valid local evidence and the
   always-defined global evidence.

Each exact reference set contains every node with a finite statistic except the
query and its immediate graph neighbors.  Target labels are first accessed by
the evaluation function after all three target scores have been constructed.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import argparse
import csv
import hashlib
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
    Config, GRAPH_IDS, STAT_NAMES, extract, inject, load_planetoid,
    preprocess, rng_for, write_csv,
)

SEEDS = [11, 23, 37, 53, 71]
TARGETS = ["cora", "citeseer"]
VARIANTS = ["exact_local", "exact_global", "exact_local_global"]


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def global_feature_radius(raw_features):
    """RMS standardized feature radius; label-free and defined for every node."""
    x = np.asarray(raw_features, dtype=np.float64).copy()
    for column in np.flatnonzero(~np.isfinite(x).all(axis=0)):
        finite = np.isfinite(x[:, column])
        x[~finite, column] = np.median(x[finite, column]) if finite.any() else 0.0
    standard_deviation = x.std(axis=0)
    retained = standard_deviation > 0
    if not retained.any():
        return np.zeros(len(x), dtype=float), retained
    standardized = (
        x[:, retained] - x[:, retained].mean(axis=0)
    ) / standard_deviation[retained]
    radius = np.sqrt(np.mean(standardized ** 2, axis=1))
    if not np.isfinite(radius).all():
        raise AssertionError("global feature radius is not finite")
    return radius, retained


def exact_marginal(statistics, adjacency):
    """Exact mid-CDF after excluding each query and all immediate neighbors."""
    statistics = np.asarray(statistics, dtype=np.float64)
    n, statistic_count = statistics.shape
    ranks = np.zeros((n, statistic_count), dtype=np.float64)
    masks = np.zeros_like(ranks)
    counts = np.zeros_like(ranks, dtype=np.int32)
    checks = 0
    for k in range(statistic_count):
        column = statistics[:, k]
        valid_ids = np.flatnonzero(np.isfinite(column))
        sorted_values = np.sort(column[valid_ids], kind="stable")
        for i in valid_ids:
            neighbors = adjacency.indices[
                adjacency.indptr[i]:adjacency.indptr[i + 1]
            ]
            excluded_ids = np.concatenate((np.asarray([i]), neighbors))
            excluded_values = column[excluded_ids]
            excluded_values = excluded_values[np.isfinite(excluded_values)]
            reference_count = len(valid_ids) - len(excluded_values)
            if reference_count <= 0:
                continue
            value = column[i]
            global_less = np.searchsorted(sorted_values, value, side="left")
            global_right = np.searchsorted(sorted_values, value, side="right")
            less = int(global_less - np.sum(excluded_values < value))
            equal = int(
                (global_right - global_less) - np.sum(excluded_values == value)
            )
            if less < 0 or equal < 0 or less + equal > reference_count:
                raise AssertionError("invalid exact reference accounting")
            ranks[i, k] = 2.0 * (less + 0.5 * equal) / reference_count - 1.0
            masks[i, k] = 1.0
            counts[i, k] = reference_count
            checks += 1
    if not np.isfinite(ranks).all() or np.max(np.abs(ranks)) > 1.0 + 1e-14:
        raise AssertionError("invalid exact ranks")
    return ranks, masks, counts, checks


def fixed_mean(evidence, mask):
    count = mask.sum(axis=1)
    return np.divide(
        (evidence * mask).sum(axis=1), count,
        out=np.zeros(len(evidence), dtype=np.float64), where=count > 0,
    )


def reproduce_graph(name, seed, data_root, input_root):
    """Reproduce injection and verify its local statistics against saved inputs."""
    base_features, base_adjacency = load_planetoid(name, data_root)
    features, adjacency, labels, anomaly_type, manifest = inject(
        base_features, base_adjacency, Config(),
        rng_for(seed, GRAPH_IDS[name], 0),
    )
    saved_manifest = json.loads(
        (input_root / f"seed_{seed}" / f"{name}_injection.json").read_text()
    )
    if manifest != saved_manifest:
        raise AssertionError(f"{name} seed {seed}: injection manifest mismatch")

    with np.load(input_root / f"seed_{seed}" / f"{name}.npz") as handle:
        cached = {key: handle[key] for key in handle.files}
    np.testing.assert_array_equal(labels, cached["labels"])
    np.testing.assert_array_equal(anomaly_type, cached["anomaly_type"])
    np.testing.assert_array_equal(np.diff(adjacency.indptr), cached["degree"])

    # Independent reproduction prevents a stale or mismatched statistic cache
    # from silently influencing this experiment.
    normalized, _, retained_local = preprocess(features)
    reproduced_statistics, reproduced_degree, _ = extract(
        normalized, adjacency, Config(), rng_for(seed, GRAPH_IDS[name], 1)
    )
    np.testing.assert_array_equal(reproduced_degree, cached["degree"])
    np.testing.assert_allclose(
        reproduced_statistics, cached["statistics"], rtol=0, atol=0,
        equal_nan=True,
    )
    return {
        "features": features,
        "adjacency": adjacency,
        "heldout_labels": labels,
        "heldout_anomaly_type": anomaly_type,
        "statistics": cached["statistics"],
        "retained_local_features": int(retained_local.sum()),
        "manifest": manifest,
    }


def construct_scores(graph):
    """Construct all target scores without accepting or reading target labels."""
    local_rank, local_mask, local_counts, local_checks = exact_marginal(
        graph["statistics"], graph["adjacency"]
    )
    radius, retained_global = global_feature_radius(graph["features"])
    global_rank, global_mask, global_counts, global_checks = exact_marginal(
        radius[:, None], graph["adjacency"]
    )
    if not np.all(global_mask == 1):
        raise AssertionError("global score did not cover every target node")

    local_evidence = np.abs(local_rank) * local_mask
    global_evidence = np.abs(global_rank)
    combined_evidence = np.concatenate((local_evidence, global_evidence), axis=1)
    combined_mask = np.concatenate((local_mask, global_mask), axis=1)
    scores = {
        "exact_local": fixed_mean(local_evidence, local_mask),
        "exact_global": global_evidence.ravel(),
        "exact_local_global": fixed_mean(combined_evidence, combined_mask),
    }
    for score in scores.values():
        if not np.isfinite(score).all() or np.min(score) < 0 or np.max(score) > 1:
            raise AssertionError("score outside [0,1]")
    arrays = {
        "local_rank": local_rank,
        "local_mask": local_mask,
        "local_counts": local_counts,
        "global_radius": radius,
        "global_rank": global_rank,
        "global_mask": global_mask,
        "global_counts": global_counts,
        **{f"score_{key}": value for key, value in scores.items()},
    }
    audit = {
        "local_exact_node_statistic_pairs": local_checks,
        "global_exact_nodes": global_checks,
        "global_coverage_fraction": float(global_mask.mean()),
        "any_local_valid_fraction": float((local_mask.sum(axis=1) > 0).mean()),
        "all_five_local_valid_fraction": float((local_mask.sum(axis=1) == 5).mean()),
        "retained_global_features": int(retained_global.sum()),
        "local_statistics_sha256": sha256_array(graph["statistics"]),
        "global_radius_sha256": sha256_array(radius),
    }
    return scores, arrays, audit


def evaluate_after_scoring(scores, heldout_labels, heldout_anomaly_type):
    """Only this function receives target labels."""
    populations = {
        "all_nodes": np.ones(len(heldout_labels), dtype=bool),
        "structural_vs_normal": heldout_anomaly_type != 2,
        "contextual_vs_normal": heldout_anomaly_type != 1,
    }
    rows = []
    for population, selected in populations.items():
        labels = heldout_labels[selected]
        for variant in VARIANTS:
            rows.append({
                "population": population,
                "variant": variant,
                "nodes": int(selected.sum()),
                "anomalies": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "auroc": float(roc_auc_score(labels, scores[variant][selected])),
                "ap": float(average_precision_score(labels, scores[variant][selected])),
            })
    return rows


def summarize(metric_rows):
    rows = []
    for target in TARGETS:
        for population in ("all_nodes", "structural_vs_normal", "contextual_vs_normal"):
            for variant in VARIANTS:
                selected = [
                    row for row in metric_rows
                    if row["target"] == target
                    and row["population"] == population
                    and row["variant"] == variant
                ]
                if not selected:
                    continue
                summary = {
                    "target": target, "population": population,
                    "variant": variant, "seeds": len(selected),
                }
                for field in ("nodes", "anomalies", "prevalence", "auroc", "ap"):
                    values = np.asarray([row[field] for row in selected], dtype=float)
                    summary[field + "_mean"] = float(values.mean())
                    summary[field + "_sd"] = (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    )
                summary["auroc_above_random_seeds"] = int(
                    sum(row["auroc"] > 0.5 for row in selected)
                )
                summary["ap_above_prevalence_seeds"] = int(
                    sum(row["ap"] > row["prevalence"] for row in selected)
                )
                rows.append(summary)
    return rows


def paired_differences(metric_rows):
    rows = []
    for target in TARGETS:
        for population in ("all_nodes", "structural_vs_normal", "contextual_vs_normal"):
            for comparator in ("exact_local", "exact_global"):
                for metric in ("auroc", "ap"):
                    differences = []
                    wins = 0
                    for seed in SEEDS:
                        selected = [
                            row for row in metric_rows
                            if row["target"] == target and row["seed"] == seed
                            and row["population"] == population
                        ]
                        values = {row["variant"]: row[metric] for row in selected}
                        difference = (
                            values["exact_local_global"] - values[comparator]
                        )
                        differences.append(difference)
                        wins += difference > 0
                    rows.append({
                        "target": target, "population": population,
                        "metric": metric,
                        "difference": f"exact_local_global_minus_{comparator}",
                        "mean": float(np.mean(differences)),
                        "sd": float(np.std(differences, ddof=1)),
                        "wins_out_of_5": int(wins),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SOURCE_ROOT / "data")
    parser.add_argument(
        "--inputs", type=Path,
        default=ROOT.parent / "condcal_citation_diagnostics" / "inputs",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = {
        "status": "frozen_before_target_results",
        "targets": TARGETS,
        "seeds": SEEDS,
        "variants": VARIANTS,
        "anomaly_protocol": Config().__dict__,
        "anomaly_fraction": Config().anomaly_fraction,
        "synthetic_types": ["structural clique", "contextual feature replacement"],
        "local_statistics": STAT_NAMES,
        "global_statistic": "RMS within-graph standardized raw-feature radius",
        "exact_calibration": "all finite-statistic nodes excluding query and its immediate graph neighbors; empirical mid-CDF",
        "evidence": "absolute signed rank; two-sided; range [0,1]",
        "combined_aggregation": "coverage-safe fixed mean over valid exact local evidence plus always-valid exact global evidence",
        "label_audit": {
            "downloaded_planetoid_parts": ["allx", "tx", "graph", "test.index"],
            "document_class_label_files_downloaded": False,
            "target_labels_passed_to_preprocess_extract_radius_or_calibration": False,
            "target_labels_used": "evaluation only after all target scores exist",
            "target_specific_feature_selection": False,
            "classifier_or_checkpoint_selection": False,
        },
        "interpretation_limit": "synthetic anomaly injection on two citation graphs; seeds are repeated perturbations, not independent domains",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    metrics, audits = [], []
    started = time.perf_counter()
    for seed in SEEDS:
        for target in TARGETS:
            target_dir = args.output / target / f"seed_{seed}"
            target_dir.mkdir(parents=True, exist_ok=True)
            graph = reproduce_graph(target, seed, args.data, args.inputs)
            scores, arrays, audit = construct_scores(graph)
            # Deliberately access held-out target labels only after score creation.
            rows = evaluate_after_scoring(
                scores, graph["heldout_labels"], graph["heldout_anomaly_type"]
            )
            for row in rows:
                row.update(seed=seed, target=target)
                metrics.append(row)

            np.savez_compressed(target_dir / "evidence_and_scores.npz", **arrays)
            write_csv(target_dir / "predictions.csv", [
                {
                    "node": i,
                    "injected_anomaly": int(graph["heldout_labels"][i]),
                    "anomaly_type": int(graph["heldout_anomaly_type"][i]),
                    "valid_local_statistics": int(arrays["local_mask"][i].sum()),
                    "global_valid": int(arrays["global_mask"][i, 0]),
                    **{variant: float(scores[variant][i]) for variant in VARIANTS},
                }
                for i in range(len(graph["heldout_labels"]))
            ])
            audits.append({
                "seed": seed, "target": target,
                "nodes": len(graph["heldout_labels"]),
                "features": graph["features"].shape[1],
                "edges_after_injection": graph["adjacency"].nnz // 2,
                "anomalies": int(graph["heldout_labels"].sum()),
                "structural_anomalies": int(
                    np.sum(graph["heldout_anomaly_type"] == 1)
                ),
                "contextual_anomalies": int(
                    np.sum(graph["heldout_anomaly_type"] == 2)
                ),
                "reproduced_prior_injection_manifest": True,
                "reproduced_prior_local_statistics_exactly": True,
                "target_scores_constructed_before_label_evaluation": True,
                "target_labels_used_for_evaluation_only": True,
                **audit,
            })
            print("RESULT " + json.dumps([
                row for row in metrics
                if row["seed"] == seed and row["target"] == target
                and row["population"] == "all_nodes"
            ]), flush=True)
            write_csv(args.output / "metrics.csv", metrics)
            write_csv(args.output / "summary.csv", summarize(metrics))
            write_csv(
                args.output / "paired_differences.csv", paired_differences(metrics)
            ) if all(
                any(row["seed"] == s and row["target"] == t for row in metrics)
                for s in SEEDS for t in TARGETS
            ) else None
            (args.output / "verification.json").write_text(
                json.dumps(audits, indent=2)
            )

    completion = {
        "status": "complete", "targets": TARGETS, "seeds": SEEDS,
        "variants": VARIANTS, "metric_rows": len(metrics),
        "expected_metric_rows": len(TARGETS) * len(SEEDS) * 3 * len(VARIANTS),
        "seconds": time.perf_counter() - started,
    }
    (args.output / "completion.json").write_text(json.dumps(completion, indent=2))
    print("COMPLETE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
