"""Post-freeze evidence-bank ablation on held-out/transfer targets.

This is an explanatory analysis only: it evaluates every already-defined
evidence bank while preserving exact neighborhood-excluded calibration and the
frozen reliability branch-max aggregation.  It must not be used to re-select
the final method after observing held-out labels.  Weibo is excluded.
"""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

from run_experiment import (
    AMAZON_ROOT,
    BASELINE,
    CANDIDATES,
    DEV_GRAPHS,
    GRAPH_IDS,
    HOLDOUT_SEEDS,
    MECHANISMS,
    REAL_IDS,
    REAL_SEEDS,
    TRANSFER_SEEDS,
    citation_instance,
    citation_labels,
    construct,
    load_amazon,
    load_reddit,
    metric,
    preprocess,
    scores,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
VARIANTS = (BASELINE, *CANDIDATES)


def main() -> None:
    output = ROOT / "heldout_evidence_ablation"
    output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((ROOT / "results" / "freeze.json").read_text())
    selected = freeze["selected_evidence_bank"]
    protocol = {
        "analysis_type": "post-freeze explanatory held-out evidence ablation",
        "eligible_for_model_selection": False,
        "frozen_selected_variant": selected,
        "calibration": "unchanged exact query-and-neighborhood-excluded midranks",
        "aggregation": "unchanged reliability-weighted contextual/structural branch maximum",
        "variants": list(VARIANTS),
        "citation_internal_holdout": {
            "datasets": list(DEV_GRAPHS),
            "seeds": list(HOLDOUT_SEEDS),
            "mechanisms": list(MECHANISMS),
        },
        "citation_transfer": {
            "datasets": ["pubmed"],
            "seeds": list(TRANSFER_SEEDS),
            "mechanisms": list(MECHANISMS),
        },
        "organic_targets": ["reddit", "amazon"],
        "excluded_target": "weibo",
        "target_labels_used_for_scoring": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows: list[dict] = []
    citation_specs = [(g, s) for g in DEV_GRAPHS for s in HOLDOUT_SEEDS]
    citation_specs += [("pubmed", s) for s in TRANSFER_SEEDS]
    for graph, seed in citation_specs:
        phase = "internal_holdout" if graph in DEV_GRAPHS else "citation_transfer"
        for mechanism in MECHANISMS:
            start = time.perf_counter()
            features, adjacency, stage = citation_instance(graph, mechanism, seed)
            bank = scores(construct(features, adjacency, seed, GRAPH_IDS[graph], stage))
            elapsed = time.perf_counter() - start
            labels = citation_labels(graph, mechanism, seed)
            for variant in VARIANTS:
                rows.append(metric(phase, graph, mechanism, seed, variant,
                                   labels, bank[variant], elapsed))
            print(f"{phase} {graph} {mechanism} seed={seed} {elapsed:.2f}s", flush=True)
            gc.collect()

    for name, loader in {"reddit": load_reddit, "amazon": load_amazon}.items():
        raw, adjacency, labels, labeled = loader(AMAZON_ROOT / "data")
        features, _, _ = preprocess(raw)
        chosen = labeled if name == "amazon" else np.ones(len(labels), dtype=bool)
        for seed in REAL_SEEDS:
            start = time.perf_counter()
            bank = scores(construct(features, adjacency, seed, REAL_IDS[name], 200))
            elapsed = time.perf_counter() - start
            for variant in VARIANTS:
                rows.append(metric("organic_labels", name, "organic_labels", seed,
                                   variant, labels[chosen], bank[variant][chosen], elapsed))
            print(f"organic_labels {name} seed={seed} {elapsed:.2f}s", flush=True)
            gc.collect()

    write_csv(output / "metrics.csv", rows)
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary)

    paired = []
    key_fields = ("phase", "dataset", "mechanism", "seed")
    by_key = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        by_key.setdefault(key, {})[row["variant"]] = row
    for key, bank in sorted(by_key.items()):
        reference = bank[selected]
        for variant, row in bank.items():
            paired.append({
                **dict(zip(key_fields, key)),
                "variant": variant,
                "reference": selected,
                "delta_auroc": row["auroc"] - reference["auroc"],
                "delta_ap": row["ap"] - reference["ap"],
            })
    write_csv(output / "paired_deltas_vs_frozen.csv", paired)
    (output / "completion.json").write_text(json.dumps({
        "complete": True,
        "metric_rows": len(rows),
        "instances": len(by_key),
        "variants": len(VARIANTS),
        "frozen_selected_variant": selected,
        "weibo_included": False,
        "post_freeze_selection_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
