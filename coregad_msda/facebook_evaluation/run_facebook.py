"""Leakage-audited Facebook evaluation for the frozen CoRe-GAD-MSDA method.

The script evaluates two unmodified public Facebook artifacts because the
published TAM table and the distributed files disagree on node/anomaly counts.
Target labels are never supplied to a scoring function and are accessed only
after every score vector for a dataset has been constructed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.io import loadmat
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
for directory in (
    WORKSPACE / "condcal_cora",
    WORKSPACE / "condcal_citation_exact_global",
    WORKSPACE / "core_multiscale_directional",
    WORKSPACE / "baseline_comparison",
):
    sys.path.insert(0, str(directory))

from condcal import Config, extract, preprocess  # noqa: E402
from run_exact_global_citation import exact_marginal  # noqa: E402
from run_four_mechanism_stress import structural_statistics  # noqa: E402
from run_experiment import residual_statistics, score_bank  # noqa: E402
from run_neighbor_div import neighbor_div_scores, projected_features  # noqa: E402
from run_freegad import freegad_scores, row_normalize_features  # noqa: E402
from run_dominant_exact import (  # noqa: E402
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_LAYERS,
    RECONSTRUCTION_WEIGHT,
    fit_score,
)


SEEDS = (11, 23, 37, 53, 71)
FACEBOOK_GRAPH_ID = 7
CONSTRUCTION_STAGE = 200
FROZEN_DIRECTIONAL_H1_TAILS = (
    "upper",
    "upper",
    "upper",
    "lower",
    "two-sided",
    "upper",
    "lower",
    "upper",
    "upper",
)
DATASETS = {
    "Facebook_TAM": ROOT / "data" / "Facebook.mat",
    "Facebook_ConAD": ROOT / "data" / "fb_conad_full.mat",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_adjacency(matrix) -> sparse.csr_matrix:
    adjacency = sparse.csr_matrix(matrix)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    adjacency.sort_indices()
    return adjacency


def load_dataset(name: str, path: Path):
    data = loadmat(path)
    if name == "Facebook_TAM":
        raw_features = data["Attributes"]
        adjacency = clean_adjacency(data["Network"])
    else:
        raw_features = data["X"]
        adjacency = clean_adjacency(data["A"])
    if sparse.issparse(raw_features):
        raw_features = raw_features.toarray()
    raw_features = np.asarray(raw_features, dtype=np.float32)
    labels = np.asarray(data["Label"]).ravel().astype(np.int8)
    assert raw_features.shape[0] == adjacency.shape[0] == len(labels)
    assert set(np.unique(labels)) == {0, 1}
    return raw_features, adjacency, labels


def core_score(raw_features, adjacency, seed):
    features, _, _ = preprocess(raw_features)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, FACEBOOK_GRAPH_ID, CONSTRUCTION_STAGE])
    )
    local, _, _ = extract(features, adjacency, Config(), rng)
    residual = residual_statistics(features, adjacency)[:, :1]
    structural, _ = structural_statistics(features, adjacency)
    statistics = np.concatenate([local, residual, structural], axis=1)
    ranks, mask, counts, checks = exact_marginal(statistics, adjacency)
    score = score_bank(
        statistics,
        ranks,
        mask,
        counts,
        tuple(range(6)),
        tuple(range(6, 9)),
        FROZEN_DIRECTIONAL_H1_TAILS,
    )
    diagnostic = {
        "finite_scores": bool(np.isfinite(score).all()),
        "mean_supported_statistics": float(mask.sum(axis=1).mean()),
        "zero_evidence_nodes": int(np.sum(mask.sum(axis=1) == 0)),
        "minimum_reference_count": int(counts[counts > 0].min())
        if np.any(counts > 0)
        else 0,
        "maximum_reference_count": int(counts.max()),
        "neighborhood_exclusion_checks": checks,
    }
    return score, diagnostic


def metrics(labels, score):
    return (
        float(roc_auc_score(labels, score)),
        float(average_precision_score(labels, score)),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(labels, score, seed=20260909, repeats=10000):
    """Stratified node bootstrap interval for one fixed score vector."""
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    aucs = np.empty(repeats)
    aps = np.empty(repeats)
    for index in range(repeats):
        sampled = np.r_[
            rng.choice(normal, len(normal), replace=True),
            rng.choice(anomaly, len(anomaly), replace=True),
        ]
        aucs[index], aps[index] = metrics(labels[sampled], score[sampled])
    return {
        "auroc_ci_low": float(np.quantile(aucs, 0.025)),
        "auroc_ci_high": float(np.quantile(aucs, 0.975)),
        "ap_ci_low": float(np.quantile(aps, 0.025)),
        "ap_ci_high": float(np.quantile(aps, 0.975)),
    }


def main() -> None:
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "post-freeze external evaluation",
        "primary_dataset": "Facebook_TAM",
        "primary_reason": "current unmodified file distributed by official TAM repository",
        "sensitivity_dataset": "Facebook_ConAD",
        "dataset_files": {name: str(path) for name, path in DATASETS.items()},
        "sha256": {name: sha256(path) for name, path in DATASETS.items()},
        "frozen_core_variant": "directional_h1",
        "calibration": "exact query-and-neighborhood-excluded midranks",
        "aggregation": "reliability-weighted contextual/structural branch maximum",
        "target_labels_used_for_scoring": False,
        "target_specific_tuning": False,
        "seeds": list(SEEDS),
        "baselines": {
            "NeighborDiv": "fixed equation-faithful reproduction; deterministic",
            "FreeGAD": "official Cora preset transferred unchanged; deterministic",
            "DOMINANT": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "reconstruction_weight": RECONSTRUCTION_WEIGHT,
            },
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn_version,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows: list[dict] = []
    diagnostics: list[dict] = []
    score_bank_by_dataset: dict[str, dict[str, list[np.ndarray]]] = {}
    labels_by_dataset: dict[str, np.ndarray] = {}
    dataset_metadata: list[dict] = []

    for dataset_index, (dataset, path) in enumerate(DATASETS.items()):
        raw_features, adjacency, labels_private = load_dataset(dataset, path)
        dataset_metadata.append(
            {
                "dataset": dataset,
                "nodes": len(labels_private),
                "adjacency_entries": int(adjacency.nnz),
                "undirected_edges": int(adjacency.nnz // 2),
                "features": int(raw_features.shape[1]),
                "anomalies": int(labels_private.sum()),
                "anomaly_rate": float(labels_private.mean()),
                "sha256": sha256(path),
            }
        )
        print(
            f"{dataset}: n={len(labels_private)} adjacency_entries={adjacency.nnz} "
            f"features={raw_features.shape[1]} anomalies={labels_private.sum()}",
            flush=True,
        )

        # Construct all label-free score vectors first.
        method_scores: dict[str, list[np.ndarray]] = {
            "CoRe-GAD-MSDA (frozen)": [],
            "NeighborDiv": [],
            "FreeGAD": [],
            "DOMINANT": [],
        }
        for seed in SEEDS:
            started = time.perf_counter()
            score, diagnostic = core_score(raw_features, adjacency, seed)
            seconds = time.perf_counter() - started
            method_scores["CoRe-GAD-MSDA (frozen)"].append(score)
            diagnostic.update(dataset=dataset, seed=seed, seconds=seconds)
            diagnostics.append(diagnostic)
            print(f"  CoRe seed={seed} t={seconds:.2f}s", flush=True)

        started = time.perf_counter()
        embedding = projected_features(raw_features)
        score, _, _, _ = neighbor_div_scores(embedding, adjacency)
        neighbor_seconds = time.perf_counter() - started
        method_scores["NeighborDiv"].append(score)
        print(f"  NeighborDiv t={neighbor_seconds:.2f}s", flush=True)

        started = time.perf_counter()
        normalized = row_normalize_features(raw_features)
        score, _, _, _ = freegad_scores(normalized, adjacency)
        freegad_seconds = time.perf_counter() - started
        method_scores["FreeGAD"].append(score)
        print(f"  FreeGAD t={freegad_seconds:.2f}s", flush=True)

        dominant_seconds = []
        for seed in SEEDS:
            started = time.perf_counter()
            score, final_loss = fit_score(raw_features, adjacency, seed)
            seconds = time.perf_counter() - started
            dominant_seconds.append(seconds)
            method_scores["DOMINANT"].append(score)
            print(
                f"  DOMINANT seed={seed} loss={final_loss:.6f} t={seconds:.2f}s",
                flush=True,
            )

        # Labels are accessed only after every method has finished scoring.
        labels = labels_private.copy()
        labels_by_dataset[dataset] = labels
        score_bank_by_dataset[dataset] = method_scores
        timings = {
            "CoRe-GAD-MSDA (frozen)": [d["seconds"] for d in diagnostics if d["dataset"] == dataset],
            "NeighborDiv": [neighbor_seconds],
            "FreeGAD": [freegad_seconds],
            "DOMINANT": dominant_seconds,
        }
        seeds = {
            "CoRe-GAD-MSDA (frozen)": list(SEEDS),
            "NeighborDiv": ["deterministic"],
            "FreeGAD": ["deterministic"],
            "DOMINANT": list(SEEDS),
        }
        for method, vectors in method_scores.items():
            for run_index, vector in enumerate(vectors):
                auroc, ap = metrics(labels, vector)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seeds[method][run_index],
                        "auroc": auroc,
                        "ap": ap,
                        "prevalence": float(labels.mean()),
                        "seconds": timings[method][run_index],
                    }
                )
                print(
                    f"  RESULT {method} seed={seeds[method][run_index]} "
                    f"AUROC={auroc:.6f} AP={ap:.6f}",
                    flush=True,
                )

    write_csv(output / "dataset_metadata.csv", dataset_metadata)
    write_csv(output / "metrics.csv", rows)
    (output / "core_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    summaries = []
    for dataset, method_scores in score_bank_by_dataset.items():
        labels = labels_by_dataset[dataset]
        for method, vectors in method_scores.items():
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            ensemble_score = np.mean(np.vstack(vectors), axis=0)
            ensemble_auroc, ensemble_ap = metrics(labels, ensemble_score)
            summary = {
                "dataset": dataset,
                "method": method,
                "runs": len(selected),
                "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
                "auroc_std": float(np.std([r["auroc"] for r in selected], ddof=0)),
                "ap_mean": float(np.mean([r["ap"] for r in selected])),
                "ap_std": float(np.std([r["ap"] for r in selected], ddof=0)),
                "mean_score_auroc": ensemble_auroc,
                "mean_score_ap": ensemble_ap,
                "seconds_mean": float(np.mean([r["seconds"] for r in selected])),
            }
            summary.update(
                bootstrap_interval(
                    labels,
                    ensemble_score,
                    seed=20260909 + len(summaries),
                )
            )
            summaries.append(summary)
    write_csv(output / "summary.csv", summaries)

    primary = [row for row in summaries if row["dataset"] == "Facebook_TAM"]
    core = next(row for row in primary if row["method"].startswith("CoRe"))
    best_baseline_auc = max(row["auroc_mean"] for row in primary if not row["method"].startswith("CoRe"))
    best_baseline_ap = max(row["ap_mean"] for row in primary if not row["method"].startswith("CoRe"))
    conclusion = {
        "primary_dataset": "Facebook_TAM",
        "favorable_by_mean_auroc": core["auroc_mean"] > best_baseline_auc,
        "favorable_by_mean_ap": core["ap_mean"] > best_baseline_ap,
        "core_auroc_mean": core["auroc_mean"],
        "best_baseline_auroc_mean": best_baseline_auc,
        "core_ap_mean": core["ap_mean"],
        "best_baseline_ap_mean": best_baseline_ap,
        "manuscript_rule": "Include as supportive only if the frozen method is competitive and dataset-version discrepancy is disclosed.",
    }
    (output / "conclusion.json").write_text(json.dumps(conclusion, indent=2))
    print("CONCLUSION", json.dumps(conclusion), flush=True)


if __name__ == "__main__":
    main()
