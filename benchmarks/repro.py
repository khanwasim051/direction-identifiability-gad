"""Re-run the frozen CoRe-GAD-MSDA evaluation (citation + Reddit) and compare to released metrics."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, csv, time, json
from pathlib import Path
PKG = Path(PKG)
sys.path.insert(0, str(PKG / "core_multiscale_directional"))
import run_experiment as R
import numpy as np

released = {}
for r in csv.DictReader(open(PKG / "results/raw/core_msda/evaluation_metrics.csv")):
    released[(r["dataset"], r["mechanism"], int(r["seed"]), r["variant"])] = (float(r["auroc"]), float(r["ap"]))

out = []
specs = [(g, s) for g in R.DEV_GRAPHS for s in R.HOLDOUT_SEEDS] + [("pubmed", s) for s in R.TRANSFER_SEEDS]
for graph, seed in specs:
    for mech in R.MECHANISMS:
        t = time.perf_counter()
        feats, adj, stage = R.citation_instance(graph, mech, seed)
        sc = R.scores(R.construct(feats, adj, seed, R.GRAPH_IDS[graph], stage))
        el = time.perf_counter() - t
        y = R.citation_labels(graph, mech, seed)
        for v in ("directional_h1", "original_twosided"):
            m = R.metric("x", graph, mech, seed, v, y, sc[v], el)
            rel = released.get((graph, mech, seed, v))
            out.append(dict(dataset=graph, mechanism=mech, seed=seed, variant=v, auroc=m["auroc"], ap=m["ap"],
                            rel_auroc=rel[0] if rel else None, rel_ap=rel[1] if rel else None, seconds=el))
        print(graph, mech, seed, round(out[-2]["auroc"], 4), out[-2]["rel_auroc"], f"{el:.1f}s", flush=True)
raw, adj, y, mask = R.load_reddit(R.AMAZON_ROOT / "data")
feats, _, _ = R.preprocess(raw)
for seed in R.REAL_SEEDS:
    t = time.perf_counter()
    sc = R.scores(R.construct(feats, adj, seed, R.REAL_IDS["reddit"], 200)); el = time.perf_counter() - t
    for v in ("directional_h1", "original_twosided"):
        m = R.metric("x", "reddit", "organic_labels", seed, v, y, sc[v], el)
        rel = released.get(("reddit", "organic_labels", seed, v))
        out.append(dict(dataset="reddit", mechanism="organic", seed=seed, variant=v, auroc=m["auroc"], ap=m["ap"],
                        rel_auroc=rel[0] if rel else None, rel_ap=rel[1] if rel else None, seconds=el))
    print("reddit", seed, round(out[-2]["auroc"], 4), out[-2]["rel_auroc"], flush=True)
with open(RES_BENCH / "repro_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
d = [abs(r["auroc"] - r["rel_auroc"]) for r in out if r["rel_auroc"] is not None] + [abs(r["ap"] - r["rel_ap"]) for r in out if r["rel_ap"] is not None]
print("MAX_ABS_DIFF", max(d), "N", len(d))
