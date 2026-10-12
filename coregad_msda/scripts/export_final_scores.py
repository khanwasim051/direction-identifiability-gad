"""Export per-node CoRe-GAD-MSDA scores for independent reviewer auditing."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
for directory in (
    ROOT / "condcal_cora",
    ROOT / "condcal_citation_exact_global",
    ROOT / "condcal_amazon",
    ROOT / "core_reliability_fusion",
    ROOT / "core_multiscale_directional",
    ROOT / "baseline_comparison",
    ROOT / "facebook_evaluation",
):
    sys.path.insert(0, str(directory))

import run_experiment as msda  # noqa: E402
import run_reliability_fusion as fusion  # noqa: E402


SEEDS = (11, 23, 37, 53, 71)
HOLDOUT_SEEDS = (53, 71)
MECHANISMS = (
    "cross_community_edges",
    "degree_preserving_rewire",
    "ego_edge_deletion",
    "hub_star_injection",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_scores(path: Path, labels, scores, evaluation_mask=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if evaluation_mask is None:
        evaluation_mask = np.ones(len(labels), dtype=bool)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node", "label", "evaluation_mask", "score"))
        for node, (label, selected, score) in enumerate(
            zip(labels, evaluation_mask, scores)
        ):
            writer.writerow((node, int(label), int(selected), f"{score:.17g}"))


def export_citation(output: Path) -> list[dict]:
    manifest = []
    specifications = [("cora", seed) for seed in HOLDOUT_SEEDS]
    specifications += [("citeseer", seed) for seed in HOLDOUT_SEEDS]
    specifications += [("pubmed", seed) for seed in SEEDS]
    for graph, seed in specifications:
        for mechanism in MECHANISMS:
            started = time.perf_counter()
            features, adjacency, stage = msda.citation_instance(graph, mechanism, seed)
            arrays = msda.construct(features, adjacency, seed, msda.GRAPH_IDS[graph], stage)
            score = msda.scores(arrays)["directional_h1"]
            labels = msda.citation_labels(graph, mechanism, seed)
            path = output / "citation" / graph / mechanism / f"seed_{seed}.csv.gz"
            write_scores(path, labels, score)
            manifest.append({
                "dataset": graph,
                "mechanism": mechanism,
                "seed": seed,
                "nodes": len(labels),
                "anomalies": int(np.sum(labels)),
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "seconds": time.perf_counter() - started,
            })
            print(f"exported {graph}/{mechanism}/seed_{seed}", flush=True)
    return manifest


def export_realworld(output: Path, data_workspace: Path) -> list[dict]:
    manifest = []
    data_dir = data_workspace / "condcal_amazon" / "data"
    loaders = {"reddit": msda.load_reddit, "amazon": msda.load_amazon}
    graph_ids = {"reddit": 4, "amazon": 6}
    for dataset, loader in loaders.items():
        raw, adjacency, labels, evaluation_mask = loader(data_dir)
        features, _, _ = msda.preprocess(raw)
        for seed in SEEDS:
            started = time.perf_counter()
            arrays = msda.construct(features, adjacency, seed, graph_ids[dataset], 200)
            score = msda.scores(arrays)["directional_h1"]
            path = output / "realworld" / dataset / f"seed_{seed}.csv.gz"
            write_scores(path, labels, score, evaluation_mask)
            manifest.append({
                "dataset": dataset,
                "mechanism": "organic_labels",
                "seed": seed,
                "nodes": len(labels),
                "evaluated_nodes": int(np.sum(evaluation_mask)),
                "anomalies_in_evaluation": int(np.sum(labels[evaluation_mask])),
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "seconds": time.perf_counter() - started,
            })
            print(f"exported {dataset}/seed_{seed}", flush=True)
    return manifest


def export_facebook(output: Path, data_workspace: Path) -> list[dict]:
    manifest = []
    datasets = {
        "Facebook_TAM": data_workspace / "facebook_evaluation" / "data" / "Facebook.mat",
        "Facebook_ConAD": data_workspace / "facebook_evaluation" / "data" / "fb_conad_full.mat",
    }
    for dataset, source_path in datasets.items():
        data = loadmat(source_path)
        if dataset == "Facebook_TAM":
            raw = data["Attributes"]
            adjacency = sparse.csr_matrix(data["Network"])
        else:
            raw = data["X"]
            adjacency = sparse.csr_matrix(data["A"])
        if sparse.issparse(raw):
            raw = raw.toarray()
        raw = np.asarray(raw, dtype=np.float32)
        labels = np.asarray(data["Label"]).ravel().astype(np.int8)
        adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        adjacency.data[:] = 1
        adjacency.sort_indices()
        features, _, _ = msda.preprocess(raw)
        for seed in SEEDS:
            started = time.perf_counter()
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7, 200]))
            local, _, _ = msda.extract(features, adjacency, msda.Config(), rng)
            residual = msda.residual_statistics(features, adjacency)[:, :1]
            structural, _ = msda.structural_statistics(features, adjacency)
            statistics = np.concatenate([local, residual, structural], axis=1)
            ranks, mask, counts, _ = msda.exact_marginal(statistics, adjacency)
            tails = (
                "upper", "upper", "upper", "lower", "two-sided",
                "upper", "lower", "upper", "upper",
            )
            score = msda.score_bank(
                statistics, ranks, mask, counts,
                tuple(range(6)), tuple(range(6, 9)), tails,
            )
            path = output / "facebook" / dataset / f"seed_{seed}.csv.gz"
            write_scores(path, labels, score)
            manifest.append({
                "dataset": dataset,
                "mechanism": "published_injected_labels",
                "seed": seed,
                "nodes": len(labels),
                "anomalies": int(np.sum(labels)),
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "seconds": time.perf_counter() - started,
            })
            print(f"exported {dataset}/seed_{seed}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("citation", "realworld", "facebook", "all"),
        default="all",
    )
    parser.add_argument(
        "--data-workspace",
        type=Path,
        default=ROOT,
        help="Workspace containing condcal_cora/data, condcal_amazon/data, and facebook_evaluation/data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "node_scores",
    )
    args = parser.parse_args()
    data_workspace = args.data_workspace.resolve()

    msda.SOURCE_ROOT = data_workspace / "condcal_cora"
    msda.AMAZON_ROOT = data_workspace / "condcal_amazon"
    fusion.CITATION_RESULTS = (
        ROOT / "condcal_citation_exact_global" / "four_mechanism_stress_results"
    )

    rows = []
    if args.scope in ("citation", "all"):
        rows.extend(export_citation(args.output))
    if args.scope in ("realworld", "all"):
        rows.extend(export_realworld(args.output, data_workspace))
    if args.scope in ("facebook", "all"):
        rows.extend(export_facebook(args.output, data_workspace))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    existing_rows = []
    if manifest_path.exists():
        existing_rows = json.loads(manifest_path.read_text()).get("files", [])
    merged = {item["file"]: item for item in existing_rows}
    merged.update({item["file"]: item for item in rows})
    manifest_path.write_text(json.dumps({
        "method": "CoRe-GAD-MSDA",
        "variant": "directional_h1",
        "weibo_included": False,
        "files": [merged[key] for key in sorted(merged)],
    }, indent=2))
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
