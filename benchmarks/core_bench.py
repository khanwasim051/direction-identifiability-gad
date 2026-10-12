"""Frozen CoRe-GAD-MSDA (directional_h1) on standard GAD benchmark files. No tuning; labels used only for metrics."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, os, csv, json, time, gc, argparse
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, str(BENCH))
PKG = str(PKG)
sys.path.insert(0, PKG + "/core_multiscale_directional")
import run_experiment as R, bench_data as B, fast_stats as F
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDS = (11, 23, 37, 53, 71)
GID = {"Reddit": 4, "Amazon-upu": 6, "Facebook": 7}
for i, n in enumerate(B.FILES):
    GID.setdefault(n, 20 + i)
VARIANTS = {"directional_h1": R.CANDIDATES["directional_h1"],
            "directional_h123": R.CANDIDATES["directional_h123"],
            "h1_no_triangle": (tuple(range(6)), (10,), R.DIRECTIONAL),
            "twosided_h1": (tuple(range(6)), tuple(range(9, 12)), R.TWO_SIDED)}

def run(name, out, struct="fast"):
    x_raw, a, y = B.load(name)
    t0 = time.perf_counter()
    x, _, _ = R.preprocess(x_raw)
    resid = R.residual_statistics(x, a)
    st = (F.structural_statistics_fast if struct == "fast" else R.structural_statistics)(x, a)[0]
    det = np.concatenate([resid, st], axis=1)
    det_r, det_m, det_c, _ = R.exact_marginal(det, a)
    shared = time.perf_counter() - t0
    rows = []; scores = {}
    for seed in SEEDS:
        t = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([seed, GID[name], 200]))
        loc, _, _ = R.extract(x, a, R.Config(), rng)
        lr, lm, lc, _ = R.exact_marginal(loc, a)
        stats = np.concatenate([loc, det], 1); ranks = np.concatenate([lr, det_r], 1)
        mask = np.concatenate([lm, det_m], 1); counts = np.concatenate([lc, det_c], 1)
        for v, spec in VARIANTS.items():
            s = R.score_bank(stats, ranks, mask, counts, *spec)
            if v == "directional_h1": scores[seed] = s.astype(np.float32)
            rows.append(dict(dataset=name, seed=seed, variant=v, nodes=len(y), anomalies=int(y.sum()),
                             prevalence=float(y.mean()), auroc=float(roc_auc_score(y, s)),
                             ap=float(average_precision_score(y, s)),
                             seconds=shared + time.perf_counter() - t if v == "directional_h1" else None))
        print(name, seed, f"AUROC={rows[-4]['auroc']:.4f} AP={rows[-4]['ap']:.4f} t={rows[-4]['seconds']:.1f}s", flush=True)
        gc.collect()
    np.savez_compressed(out / f"scores_{name}.npz", labels=y, **{f"seed_{k}": v for k, v in scores.items()})
    return rows

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("datasets", nargs="+"); p.add_argument("--out", default=str(RES_BENCH / "core_out"))
    p.add_argument("--struct", default="fast"); a = p.parse_args()
    out = __import__("pathlib").Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets:
        rows = run(name, out, a.struct)
        with open(out / f"metrics_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
