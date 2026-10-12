"""Run protocol-compatible baselines on organic Reddit and Amazon labels.

Weibo is intentionally excluded.  Labels and Amazon's evaluation mask are
never passed to a scoring function; they are used only after each score vector
has been constructed.  NeighborDiv and FreeGAD are deterministic.  DOMINANT
is trained with five fixed random seeds and no target-label tuning.
"""

from __future__ import annotations

import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DATA_DIR = WORKSPACE / "condcal_amazon" / "data"
sys.path.insert(0, str(WORKSPACE / "condcal_amazon"))
sys.path.insert(0, str(WORKSPACE / "baseline_comparison"))

from dataset_loaders import load_amazon, load_reddit  # noqa: E402
from run_neighbor_div import (  # noqa: E402
    PROJECTION_RANK,
    SVD_RANDOM_STATE,
    neighbor_div_scores,
    projected_features,
)
from run_freegad import (  # noqa: E402
    ALPHA,
    BETA,
    NUM_ANCHORS,
    NUM_HOPS,
    freegad_scores,
    row_normalize_features,
)
from run_dominant_exact import (  # noqa: E402
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_LAYERS,
    OFFICIAL_COMMIT as DOMINANT_COMMIT,
    RECONSTRUCTION_WEIGHT,
    fit_score,
)


SEEDS = [11, 23, 37, 53, 71]
LOADERS = {"Reddit": load_reddit, "Amazon": load_amazon}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(scores, labels, mask):
    selected_scores = np.asarray(scores)[mask]
    selected_labels = np.asarray(labels)[mask]
    assert np.isfinite(selected_scores).all()
    return (
        float(roc_auc_score(selected_labels, selected_scores)),
        float(average_precision_score(selected_labels, selected_scores)),
    )


def main() -> None:
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)

    protocol = {
        "datasets": ["Reddit", "Amazon"],
        "excluded_dataset": "Weibo",
        "labels": "organic benchmark labels; evaluation only",
        "amazon_evaluation_mask": "nodes 3305:11944, matching the CARE-GNN/PyGOD convention",
        "target_specific_hyperparameter_tuning": False,
        "deterministic_methods": ["NeighborDiv", "FreeGAD"],
        "stochastic_method_seeds": {"DOMINANT": SEEDS},
        "NeighborDiv": {
            "implementation": "equation-faithful reproduction",
            "projection_rank": PROJECTION_RANK,
            "svd_random_state": SVD_RANDOM_STATE,
            "neighbor_pairs": "exhaustive unordered pairs",
        },
        "FreeGAD": {
            "implementation": "equation-faithful reproduction of official scoring equations",
            "alpha": ALPHA,
            "beta": BETA,
            "num_hops": NUM_HOPS,
            "num_anchors": NUM_ANCHORS,
            "preset": "official Cora preset transferred unchanged",
        },
        "DOMINANT": {
            "implementation": "official PyGOD layers with exact sparse structure-error identity",
            "official_commit": DOMINANT_COMMIT,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "reconstruction_weight": RECONSTRUCTION_WEIGHT,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import torch
        import torch_geometric
        import pygod

        protocol.update(
            torch=torch.__version__,
            torch_geometric=torch_geometric.__version__,
            pygod=pygod.__version__,
        )
    except Exception as exc:  # pragma: no cover
        protocol["torch_environment_error"] = repr(exc)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows: list[dict] = []
    for dataset, loader in LOADERS.items():
        raw_features, adjacency, labels, evaluation_mask = loader(DATA_DIR)
        prevalence = float(np.mean(labels[evaluation_mask]))
        print(
            f"{dataset}: n={len(labels)} edges={adjacency.nnz // 2} "
            f"eval={int(evaluation_mask.sum())} prevalence={prevalence:.6f}",
            flush=True,
        )

        start = time.perf_counter()
        embedding = projected_features(raw_features)
        score, _, valid, _ = neighbor_div_scores(embedding, adjacency)
        seconds = time.perf_counter() - start
        auroc, ap = evaluate(score, labels, evaluation_mask)
        rows.append({
            "method": "NeighborDiv",
            "dataset": dataset,
            "seed": "deterministic",
            "auroc": auroc,
            "ap": ap,
            "prevalence": prevalence,
            "seconds": seconds,
            "valid_scored_nodes": int(valid.sum()),
            "epochs": 0,
            "trainable_parameters": 0,
        })
        print(f"  NeighborDiv AUROC={auroc:.6f} AP={ap:.6f} t={seconds:.2f}s", flush=True)
        del embedding, score, valid
        gc.collect()

        start = time.perf_counter()
        normalized = row_normalize_features(raw_features)
        score, _, _, _ = freegad_scores(normalized, adjacency)
        seconds = time.perf_counter() - start
        auroc, ap = evaluate(score, labels, evaluation_mask)
        rows.append({
            "method": "FreeGAD",
            "dataset": dataset,
            "seed": "deterministic",
            "auroc": auroc,
            "ap": ap,
            "prevalence": prevalence,
            "seconds": seconds,
            "valid_scored_nodes": int(evaluation_mask.sum()),
            "epochs": 0,
            "trainable_parameters": 0,
        })
        print(f"  FreeGAD AUROC={auroc:.6f} AP={ap:.6f} t={seconds:.2f}s", flush=True)
        del normalized, score
        gc.collect()

        for seed in SEEDS:
            start = time.perf_counter()
            score, final_loss = fit_score(raw_features, adjacency, seed)
            seconds = time.perf_counter() - start
            auroc, ap = evaluate(score, labels, evaluation_mask)
            rows.append({
                "method": "DOMINANT",
                "dataset": dataset,
                "seed": seed,
                "auroc": auroc,
                "ap": ap,
                "prevalence": prevalence,
                "seconds": seconds,
                "valid_scored_nodes": int(evaluation_mask.sum()),
                "epochs": EPOCHS,
                "trainable_parameters": "model-dependent; recorded separately",
                "final_loss": final_loss,
            })
            print(
                f"  DOMINANT seed={seed} AUROC={auroc:.6f} "
                f"AP={ap:.6f} t={seconds:.2f}s",
                flush=True,
            )
            del score
            gc.collect()

    write_csv(output / "metrics.csv", rows)
    summary: list[dict] = []
    for method in ("NeighborDiv", "FreeGAD", "DOMINANT"):
        for dataset in LOADERS:
            selected = [r for r in rows if r["method"] == method and r["dataset"] == dataset]
            summary.append({
                "method": method,
                "dataset": dataset,
                "runs": len(selected),
                "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
                "auroc_std": float(np.std([r["auroc"] for r in selected], ddof=0)),
                "ap_mean": float(np.mean([r["ap"] for r in selected])),
                "ap_std": float(np.std([r["ap"] for r in selected], ddof=0)),
                "seconds_mean": float(np.mean([r["seconds"] for r in selected])),
            })
    write_csv(output / "summary.csv", summary)
    (output / "completion.json").write_text(json.dumps({"complete": True, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
