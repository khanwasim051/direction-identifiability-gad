"""Computes the base detector's relational statistics, neighbourhood-excluded ranks, masks and reference counts for
every graph and seed (5 seeds), and stores them in WORK/cache. Labels are saved separately; the labels of the four
held-out graphs go to WORK/sealed_labels.
Usage: python gad/statistics_cache.py Cora CiteSeer ...   (graph names as in gad/bench_data.py)"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import WORK, PKG, GADLIB, CACHE, SEALED  # noqa: E402
import sys, os, time, numpy as np
sys.path.insert(0, str(GADLIB))
PKG = str(PKG); sys.path.insert(0, PKG + "/core_multiscale_directional")
import run_experiment as R, bench_data as B, fast_stats as F
OUT = str(CACHE); SEALED = str(SEALED)
os.makedirs(OUT, exist_ok=True); os.makedirs(SEALED, exist_ok=True)
T2 = {"CS", "Photo", "Books", "Enron"}
GID = {"Reddit": 4, "Facebook": 7}
for i, n in enumerate(list(B.FILES) + list(B.PYGOD)): GID.setdefault(n, 20 + i)
COLS = list(range(6)) + [9, 10, 11]          # directional_h1 bank: 6 contextual + 3 structural
def save(tag, stats, ranks, mask, counts):
    np.savez_compressed(f"{OUT}/{tag}.npz", stats=stats[:, COLS], ranks=ranks[:, COLS], mask=mask[:, COLS], counts=counts[:, COLS])
def standard(name):
    x_raw, a, y = B.load_any(name)
    np.save(f"{SEALED if name in T2 else OUT}/labels_{name}.npy", y); del y
    x, _, _ = R.preprocess(x_raw)
    det = np.concatenate([R.residual_statistics(x, a), F.structural_statistics_fast(x, a)[0]], 1)
    dr, dm, dc, _ = R.exact_marginal(det, a)
    for seed in (11, 23, 37, 53, 71):
        if os.path.exists(f"{OUT}/{name}__{seed}.npz"): continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, GID[name], 200]))
        loc, _, _ = R.extract(x, a, R.Config(), rng); lr, lm, lc, _ = R.exact_marginal(loc, a)
        save(f"{name}__{seed}", np.c_[loc, det], np.c_[lr, dr], np.c_[lm, dm], np.c_[lc, dc])
    print("done", name, flush=True)
if __name__ == "__main__":
    for n in sys.argv[1:]:
        t = time.perf_counter(); standard(n); print(n, f"{time.perf_counter()-t:.0f}s", flush=True)
