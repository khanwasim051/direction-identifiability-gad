"""Final multi-scale, direction-aware evidence experiment for CoRe-GAD."""

import argparse, csv, gc, json, os, platform, resource, sys, time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy
from scipy.stats import rankdata
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE_ROOT = WORKSPACE / "condcal_cora"
CITATION_ROOT = WORKSPACE / "condcal_citation_exact_global"
AMAZON_ROOT = WORKSPACE / "condcal_amazon"
FUSION_ROOT = WORKSPACE / "core_reliability_fusion"
for path in (SOURCE_ROOT, CITATION_ROOT, AMAZON_ROOT, FUSION_ROOT):
    sys.path.insert(0, str(path))

from condcal import Config, GRAPH_IDS, extract, load_planetoid, preprocess, rng_for  # noqa: E402
from run_exact_global_citation import exact_marginal  # noqa: E402
from run_four_mechanism_stress import (  # noqa: E402
    ANOMALY_FRACTION, INJECTORS, MECHANISMS, structural_statistics,
)
from dataset_loaders import load_amazon, load_reddit  # noqa: E402
from run_reliability_fusion import citation_labels  # noqa: E402

DEV_GRAPHS = ("cora", "citeseer")
DEV_SEEDS = (11, 23, 37)
HOLDOUT_SEEDS = (53, 71)
TRANSFER_SEEDS = (11, 23, 37, 53, 71)
REAL_SEEDS = TRANSFER_SEEDS
REAL_IDS = {"reddit": 4, "amazon": 6}
NAMES = (
    "node_neighbor_discrepancy", "neighbor_pair_mean", "neighbor_pair_variance",
    "closure", "neighbor_degree", "propagation_residual_h1",
    "propagation_residual_h2", "propagation_residual_h3",
    "degree_context_residual", "ego_triangle_load", "unexpected_edge_tail",
    "motif_incoherence",
)
DIRECTIONAL = (
    "upper", "upper", "upper", "lower", "two-sided",
    "upper", "upper", "upper", "upper", "lower", "upper", "upper",
)
TWO_SIDED = ("two-sided",) * len(NAMES)
BASELINE = "original_twosided"
CANDIDATES = {
    "directional_original": (tuple(range(5)), tuple(range(9, 12)), DIRECTIONAL),
    "directional_h1": (tuple(range(6)), tuple(range(9, 12)), DIRECTIONAL),
    "directional_h12": (tuple(range(7)), tuple(range(9, 12)), DIRECTIONAL),
    "directional_h123": (tuple(range(8)), tuple(range(9, 12)), DIRECTIONAL),
    "directional_h123_degree": (tuple(range(9)), tuple(range(9, 12)), DIRECTIONAL),
    "twosided_h123_degree": (tuple(range(9)), tuple(range(9, 12)), TWO_SIDED),
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def masked_mean(values, weights):
    denominator = weights.sum(axis=1)
    return np.divide((values * weights).sum(axis=1), denominator,
                     out=np.zeros(len(values)), where=denominator > 0)


def marginal_ranks(statistics):
    output = np.zeros_like(statistics, dtype=float)
    for k in range(statistics.shape[1]):
        valid = np.isfinite(statistics[:, k]); count = int(valid.sum())
        if count:
            output[valid, k] = 2 * (
                rankdata(statistics[valid, k], method="average") - 0.5
            ) / count - 1
    return output


def orient(ranks, mask, tails):
    evidence = np.zeros_like(ranks, dtype=float)
    for k, tail in enumerate(tails):
        if tail == "upper": evidence[:, k] = (ranks[:, k] + 1) / 2
        elif tail == "lower": evidence[:, k] = (1 - ranks[:, k]) / 2
        else: evidence[:, k] = np.abs(ranks[:, k])
    return evidence * mask


def score_bank(statistics, ranks, mask, counts, local_ids, structural_ids, tails):
    evidence = orient(ranks, mask, tails)
    valid_count = mask.sum(axis=0)
    support = np.clip(counts / np.maximum(valid_count - 1, 1)[None, :], 0, 1) * mask
    dispersion = np.zeros(statistics.shape[1])
    for k in range(statistics.shape[1]):
        valid = mask[:, k] > 0
        if valid.any():
            q1, q3 = np.quantile(evidence[valid, k], (0.25, 0.75))
            dispersion[k] = np.clip((q3 - q1) / 0.5, 0, 1)
    stability = np.clip(1 - np.abs(ranks - marginal_ranks(statistics)) / 2, 0, 1) * mask
    weights = support * dispersion[None, :] * stability
    contextual = masked_mean(evidence[:, local_ids], weights[:, local_ids])
    structural = masked_mean(evidence[:, structural_ids], weights[:, structural_ids])
    return np.maximum(contextual, structural)


def residual_statistics(features, adjacency):
    x = np.asarray(features, dtype=np.float32)
    degree = np.diff(adjacency.indptr).astype(float)
    propagated = x.copy(); columns = []
    for _ in range(3):
        propagated = np.asarray(
            (adjacency @ propagated + propagated) / (degree + 1)[:, None]
        )
        norm = np.linalg.norm(propagated, axis=1)
        cosine = np.divide(np.einsum("ij,ij->i", x, propagated), norm,
                           out=np.zeros(len(x)), where=norm > 0)
        columns.append(np.clip(1 - cosine, 0, 2))
    log_degree = np.log1p(degree)
    neighbor_mean = np.divide(adjacency @ log_degree, degree,
                              out=np.full(len(x), np.nan), where=degree > 0)
    columns.append(np.abs(log_degree - neighbor_mean))
    return np.column_stack(columns)


def construct(features, adjacency, seed, graph_id, stage):
    rng = np.random.default_rng(np.random.SeedSequence([seed, graph_id, stage]))
    local, _, _ = extract(features, adjacency, Config(), rng)
    structural, _ = structural_statistics(features, adjacency)
    statistics = np.concatenate(
        [local, residual_statistics(features, adjacency), structural], axis=1
    )
    ranks, mask, counts, checks = exact_marginal(statistics, adjacency)
    return statistics, ranks, mask, counts, checks


def scores(arrays):
    statistics, ranks, mask, counts = arrays[:4]
    result = {BASELINE: score_bank(
        statistics, ranks, mask, counts, tuple(range(5)), tuple(range(9, 12)), TWO_SIDED
    )}
    for name, specification in CANDIDATES.items():
        result[name] = score_bank(statistics, ranks, mask, counts, *specification)
    return result


def citation_instance(graph, mechanism, seed):
    raw, base = load_planetoid(graph, SOURCE_ROOT / "data")
    features, _, _ = preprocess(raw)
    mechanism_id = MECHANISMS.index(mechanism)
    adjacency, _, _, _ = INJECTORS[mechanism](
        base, features, int(np.floor(ANOMALY_FRACTION * len(features))),
        rng_for(seed, GRAPH_IDS[graph], 100 + mechanism_id),
    )
    adjacency.eliminate_zeros(); adjacency.sort_indices()
    return features, adjacency, 200 + mechanism_id


def metric(phase, dataset, mechanism, seed, variant, labels, score, seconds):
    return {"phase": phase, "dataset": dataset, "mechanism": mechanism,
            "seed": seed, "variant": variant, "nodes": len(labels),
            "anomalies": int(labels.sum()), "prevalence": float(labels.mean()),
            "auroc": float(roc_auc_score(labels, score)),
            "ap": float(average_precision_score(labels, score)),
            "construction_seconds": seconds}


def summarize(rows):
    output = []
    for key in sorted({(r["phase"], r["dataset"], r["variant"]) for r in rows}):
        selected = [r for r in rows if (r["phase"], r["dataset"], r["variant"]) == key]
        output.append({"phase": key[0], "dataset": key[1], "variant": key[2],
                       "runs": len(selected),
                       "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
                       "auroc_std": float(np.std([r["auroc"] for r in selected], ddof=1)),
                       "ap_mean": float(np.mean([r["ap"] for r in selected])),
                       "ap_std": float(np.std([r["ap"] for r in selected], ddof=1)),
                       "seconds_mean": float(np.mean([r["construction_seconds"] for r in selected]))})
    return output


def run_development(output):
    protocol = {
        "status": "evidence development; calibration and aggregation fixed",
        "calibration": "exact query-and-neighborhood-excluded midranks",
        "aggregation": "reliability-weighted contextual/structural branch maximum",
        "development_graphs": DEV_GRAPHS, "development_seeds": DEV_SEEDS,
        "mechanisms": MECHANISMS, "candidate_banks": list(CANDIDATES),
        "nonselectable_reference": BASELINE,
        "tail_directions": dict(zip(NAMES, DIRECTIONAL)),
        "selection": "mean prevalence-adjusted AUROC/AP utility",
        "evaluation_not_accessed_during_selection": True,
    }
    (output / "development_protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    for graph in DEV_GRAPHS:
        for mechanism in MECHANISMS:
            for seed in DEV_SEEDS:
                start = time.perf_counter()
                features, adjacency, stage = citation_instance(graph, mechanism, seed)
                run_scores = scores(construct(features, adjacency, seed, GRAPH_IDS[graph], stage))
                elapsed = time.perf_counter() - start
                labels = citation_labels(graph, mechanism, seed)  # accessed after scores
                for variant, score in run_scores.items():
                    rows.append(metric("development", graph, mechanism, seed,
                                       variant, labels, score, elapsed))
                gc.collect()
    write_csv(output / "development_metrics.csv", rows)
    summary = []
    for variant in (BASELINE, *CANDIDATES):
        selected = [r for r in rows if r["variant"] == variant]
        utility = np.mean([0.5 * ((r["auroc"] - 0.5) / 0.5 +
            (r["ap"] - r["prevalence"]) / (1 - r["prevalence"])) for r in selected])
        summary.append({"variant": variant, "eligible": variant != BASELINE,
                        "runs": len(selected), "selection_utility": float(utility),
                        "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
                        "ap_mean": float(np.mean([r["ap"] for r in selected]))})
    summary.sort(key=lambda r: (-r["selection_utility"], r["variant"]))
    write_csv(output / "development_summary.csv", summary)
    selected = next(r["variant"] for r in summary if r["eligible"])
    freeze = {"status": "frozen_before_evaluation", "selected_evidence_bank": selected,
              "fixed_aggregation": "reliability_branch_max",
              "candidate_order": [r["variant"] for r in summary if r["eligible"]],
              "evaluation_variants": [selected, BASELINE]}
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2))
    return selected


def run_evaluation(output):
    freeze = json.loads((output / "freeze.json").read_text())
    if freeze["status"] != "frozen_before_evaluation": raise RuntimeError("not frozen")
    selected = freeze["selected_evidence_bank"]; rows = []; start_all = time.perf_counter()
    specifications = [(g, s) for g in DEV_GRAPHS for s in HOLDOUT_SEEDS]
    specifications += [("pubmed", s) for s in TRANSFER_SEEDS]
    for graph, seed in specifications:
        phase = "internal_holdout" if graph in DEV_GRAPHS else "citation_transfer"
        for mechanism in MECHANISMS:
            start = time.perf_counter()
            features, adjacency, stage = citation_instance(graph, mechanism, seed)
            run_scores = scores(construct(features, adjacency, seed, GRAPH_IDS[graph], stage))
            elapsed = time.perf_counter() - start
            labels = citation_labels(graph, mechanism, seed)  # accessed after scores
            for variant in (selected, BASELINE):
                rows.append(metric(phase, graph, mechanism, seed, variant,
                                   labels, run_scores[variant], elapsed))
            gc.collect()
    for name, loader in {"reddit": load_reddit, "amazon": load_amazon}.items():
        raw, adjacency, labels, labeled = loader(AMAZON_ROOT / "data")
        features, _, _ = preprocess(raw)
        chosen = labeled if name == "amazon" else np.ones(len(labels), dtype=bool)
        for seed in REAL_SEEDS:
            start = time.perf_counter()
            run_scores = scores(construct(features, adjacency, seed, REAL_IDS[name], 200))
            elapsed = time.perf_counter() - start
            for variant in (selected, BASELINE):
                rows.append(metric("organic_labels", name, "organic_labels", seed,
                                   variant, labels[chosen], run_scores[variant][chosen], elapsed))
            gc.collect()
    write_csv(output / "evaluation_metrics.csv", rows)
    write_csv(output / "evaluation_summary.csv", summarize(rows))
    completion = {"status": "complete", "selected_evidence_bank": selected,
                  "evaluation_metric_rows": len(rows),
                  "target_labels_used_for_scoring": False,
                  "trainable_parameters": 0, "training_required": False,
                  "elapsed_seconds": time.perf_counter() - start_all,
                  "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "sklearn": sklearn_version}
    (output / "evaluation_completion.json").write_text(json.dumps(completion, indent=2))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "evaluation", "all"), nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("development", "all"): print("FROZEN", run_development(args.output), flush=True)
    if args.phase in ("evaluation", "all"): print("COMPLETE", run_evaluation(args.output), flush=True)


if __name__ == "__main__": main()
