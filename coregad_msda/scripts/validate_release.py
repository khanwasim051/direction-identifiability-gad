"""Validate release completeness, headline values, and SHA-256 integrity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(relative_path: str) -> list[dict]:
    with (ROOT / relative_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: str, expected: float, tolerance: float = 5e-5) -> None:
    value = float(actual)
    assert math.isfinite(value)
    assert abs(value - expected) <= tolerance, (value, expected)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    protocol = json.loads((ROOT / "configs/final_protocol.json").read_text())
    assert protocol["selected_evidence_bank"] == "directional_h1"
    assert protocol["excluded_dataset"] == "weibo"
    assert protocol["target_labels_used_for_scoring"] is False
    assert protocol["trainable_parameters"] == 0

    citation = rows("results/manuscript/table_citation_comparison.csv")
    expected_citation = {
        "cora": (0.7575644873400411, 0.1800995896696745),
        "citeseer": (0.7797451670014446, 0.25389300952032107),
        "pubmed": (0.8160727604761145, 0.2985180161925626),
    }
    for dataset, expected in expected_citation.items():
        row = next(r for r in citation if r["method"] == "CoRe-GAD-MSDA" and r["dataset"] == dataset)
        close(row["auroc"], expected[0])
        close(row["ap"], expected[1])

    realworld = rows("results/manuscript/table_reddit_amazon.csv")
    for dataset, expected in {
        "Reddit": (0.5262789396704431, 0.0344789578824513),
        "Amazon": (0.7375821404678732, 0.1878506074635612),
    }.items():
        row = next(r for r in realworld if r["method"] == "CoRe-GAD-MSDA (frozen)" and r["dataset"] == dataset)
        close(row["auroc_mean"], expected[0])
        close(row["ap_mean"], expected[1])

    facebook = rows("results/manuscript/table_facebook.csv")
    row = next(r for r in facebook if r["dataset"] == "Facebook_TAM" and r["method"] == "CoRe-GAD-MSDA (frozen)")
    close(row["auroc_mean"], 0.87775)
    close(row["ap_mean"], 0.1938783472520267)

    for path in (ROOT / "results" / "manuscript").glob("*.csv"):
        assert "weibo" not in path.read_text().lower(), path

    score_manifest = ROOT / "results" / "node_scores" / "manifest.json"
    if score_manifest.exists():
        manifest = json.loads(score_manifest.read_text())
        assert manifest["weibo_included"] is False
        assert len(manifest["files"]) == 56
        for item in manifest["files"]:
            path = ROOT / item["file"]
            assert path.is_file(), path
            assert digest(path) == item["sha256"], path

        recomputed = rows("results/node_scores/recomputed_metrics.csv")
        score_expectations = {
            "cora": (0.7575644873400411, 0.1800995896696745),
            "citeseer": (0.7797451670014446, 0.25389300952032107),
            "pubmed": (0.8160727604761145, 0.2985180161925626),
            "reddit": (0.5262789396704431, 0.034478957882451367),
            "amazon": (0.7375821404678732, 0.18785060746356128),
            "Facebook_TAM": (0.87775, 0.19387834725202674),
            "Facebook_ConAD": (0.8839226384080019, 0.22645052025815446),
        }
        for dataset, expected in score_expectations.items():
            selected = [r for r in recomputed if r["dataset"] == dataset]
            close(str(sum(float(r["auroc"]) for r in selected) / len(selected)), expected[0])
            close(str(sum(float(r["ap"]) for r in selected) / len(selected)), expected[1])

    required = [
        "README.md",
        "DATASETS.md",
        "RESULTS_INDEX.md",
        "configs/final_protocol.json",
        "configs/frozen_evidence_bank.json",
        "configs/frozen_aggregation_rule.json",
        "results/manuscript/table_mechanism_stress.csv",
        "results/manuscript/table_efficiency.csv",
        "scripts/export_final_scores.py",
        "scripts/recompute_metrics.py",
    ]
    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path

    print("Release validation: PASS")


if __name__ == "__main__":
    main()
