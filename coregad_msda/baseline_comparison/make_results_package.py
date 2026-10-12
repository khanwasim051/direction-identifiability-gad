"""Validate completed runs and build manuscript-facing comparison artifacts."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE_METRICS = (
    ROOT.parent / "condcal_citation_exact_global"
    / "four_mechanism_stress_results" / "metrics.csv"
)
GRAPHS = ["cora", "citeseer", "pubmed"]
MECHANISMS = [
    "cross_community_edges",
    "degree_preserving_rewire",
    "ego_edge_deletion",
    "hub_star_injection",
]
SEEDS = [11, 23, 37, 53, 71]
COMPLETED = {
    "NeighborDiv": ROOT / "results" / "neighbor_div" / "metrics.csv",
    "FreeGAD": ROOT / "results" / "freegad" / "metrics.csv",
    "DOMINANT": ROOT / "results" / "dominant" / "metrics.csv",
}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_rows(method, path):
    rows = read_rows(path)
    output = []
    for row in rows:
        output.append({
            "method": method,
            "graph": row["graph"],
            "mechanism": row["mechanism"],
            "seed": int(row["seed"]),
            "auroc": float(row["auroc"]),
            "ap": float(row["ap"]),
        })
    return output


def main():
    all_rows = {}
    core = []
    for row in read_rows(CORE_METRICS):
        if row["population"] == "all_nodes" and row["variant"] == "structural_fixed_mean":
            core.append({
                "method": "CoRe-GAD",
                "graph": row["graph"],
                "mechanism": row["mechanism"],
                "seed": int(row["seed"]),
                "auroc": float(row["auroc"]),
                "ap": float(row["ap"]),
            })
    all_rows["CoRe-GAD"] = core
    for method, path in COMPLETED.items():
        all_rows[method] = numeric_rows(method, path)

    expected_keys = {
        (graph, mechanism, seed)
        for graph in GRAPHS
        for mechanism in MECHANISMS
        for seed in SEEDS
    }
    validation = {"expected_keys": len(expected_keys), "methods": {}}
    for method, rows in all_rows.items():
        keys = {(r["graph"], r["mechanism"], r["seed"]) for r in rows}
        values = np.asarray([[r["auroc"], r["ap"]] for r in rows])
        validation["methods"][method] = {
            "rows": len(rows),
            "unique_keys": len(keys),
            "missing_keys": [list(key) for key in sorted(expected_keys - keys)],
            "extra_keys": [list(key) for key in sorted(keys - expected_keys)],
            "all_metrics_finite": bool(np.isfinite(values).all()),
            "all_metrics_in_unit_interval": bool(
                ((values >= 0.0) & (values <= 1.0)).all()
            ),
        }
        assert keys == expected_keys
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()

    detailed = []
    main_table = []
    for method in ["NeighborDiv", "FreeGAD", "DOMINANT", "CoRe-GAD"]:
        rows = all_rows[method]
        record = {"Method": method}
        for graph in GRAPHS:
            selected = [r for r in rows if r["graph"] == graph]
            record[f"{graph}_auroc"] = float(np.mean([r["auroc"] for r in selected]))
            record[f"{graph}_ap"] = float(np.mean([r["ap"] for r in selected]))
            for mechanism in MECHANISMS:
                subset = [r for r in selected if r["mechanism"] == mechanism]
                detailed.append({
                    "Method": method,
                    "Graph": graph,
                    "Mechanism": mechanism,
                    "Runs": len(subset),
                    "AUROC_mean": float(np.mean([r["auroc"] for r in subset])),
                    "AUROC_std": float(np.std([r["auroc"] for r in subset])),
                    "AP_mean": float(np.mean([r["ap"] for r in subset])),
                    "AP_std": float(np.std([r["ap"] for r in subset])),
                })
        record["overall_auroc"] = float(np.mean([r["auroc"] for r in rows]))
        record["overall_ap"] = float(np.mean([r["ap"] for r in rows]))
        main_table.append(record)

    write_csv(ROOT / "completed_comparison_table.csv", main_table)
    write_csv(ROOT / "mechanism_level_results.csv", detailed)
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2))

    headers = [
        "Method", "Cora AUROC", "Cora AP", "CiteSeer AUROC", "CiteSeer AP",
        "PubMed AUROC", "PubMed AP", "Overall AUROC", "Overall AP",
    ]
    md = [
        "# Controlled four-mechanism comparison",
        "",
        "All numeric entries below are measured results averaged over four mechanisms and five perturbation/training seeds (20 runs per graph).",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * 8) + "|",
    ]
    for row in main_table:
        md.append(
            "| " + " | ".join([
                row["Method"],
                f"{row['cora_auroc']:.4f}", f"{row['cora_ap']:.4f}",
                f"{row['citeseer_auroc']:.4f}", f"{row['citeseer_ap']:.4f}",
                f"{row['pubmed_auroc']:.4f}", f"{row['pubmed_ap']:.4f}",
                f"{row['overall_auroc']:.4f}", f"{row['overall_ap']:.4f}",
            ]) + " |"
        )
    md.extend([
        "",
        "## Baselines not entered as numeric rows",
        "",
        "- **ReFi-GAD:** its official test routine samples labeled normal and anomalous support nodes from the target graph (50 of each in its released configuration). It is therefore a few-shot comparator, not a label-free zero-shot comparator. A value from that routine must not be placed in the same table without an explicit few-shot marker and protocol column.",
        "- **AnomalyGFM:** its official zero-shot code trains a model on a labeled source graph before target inference. The repository provides neither a citation-ready pretrained zero-shot checkpoint nor a uniquely determined source graph for this experiment. A source-domain protocol must be declared before running it; otherwise the result is under-specified.",
        "",
        "## Interpretation constraint",
        "",
        "DOMINANT is the strongest completed method in the aggregate table. These experiments therefore do not support a state-of-the-art performance claim for CoRe-GAD. They can support claims about label-free exact calibration, leakage control, and behavior across distinct structural mechanisms, provided those claims are stated separately from comparative superiority.",
        "",
        "## Reproduction notes",
        "",
        "- NeighborDiv: equation-faithful independent reproduction, rank 8, exhaustive unordered neighbor pairs, no target tuning.",
        "- FreeGAD: equation-faithful reproduction of official commit `c1b007d69174e22fcf9d7ff7fea0dcb88ebdf2a0`; official Cora preset transferred unchanged to all citation graphs. A direct Torch cross-check gave maximum embedding error `2.98e-8` and maximum score error `2.55e-5`.",
        "- DOMINANT: official PyGOD layers from commit `c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8`, fixed 20 epochs, with an algebraically exact sparse structure-error calculation. Dense-versus-sparse validation gave maximum score error `1.91e-6`.",
        "- CoRe-GAD: frozen `structural_fixed_mean` results from the leakage-audited four-mechanism stress test.",
    ])
    (ROOT / "RESULTS_README.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
