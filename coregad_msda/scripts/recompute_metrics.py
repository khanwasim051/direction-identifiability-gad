"""Recompute AUROC and AP solely from the exported reviewer score files."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = ROOT / "results" / "node_scores"


def load_scores(path: Path):
    labels, masks, scores = [], [], []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row["label"]))
            masks.append(bool(int(row["evaluation_mask"])))
            scores.append(float(row["score"]))
    return np.asarray(labels), np.asarray(masks), np.asarray(scores)


def main() -> None:
    manifest = json.loads((SCORE_ROOT / "manifest.json").read_text())
    rows = []
    for item in manifest["files"]:
        labels, mask, scores = load_scores(ROOT / item["file"])
        selected_labels = labels[mask]
        selected_scores = scores[mask]
        rows.append({
            "dataset": item["dataset"],
            "mechanism": item["mechanism"],
            "seed": item["seed"],
            "nodes": len(labels),
            "evaluated_nodes": int(mask.sum()),
            "anomalies": int(selected_labels.sum()),
            "auroc": float(roc_auc_score(selected_labels, selected_scores)),
            "ap": float(average_precision_score(selected_labels, selected_scores)),
        })

    output = SCORE_ROOT / "recomputed_metrics.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
