"""Revision: statistics for the 15 extra seeds 101-115 (same pipeline as gad/statistics_cache.py).
Usage: python fdc/rev_extra_seeds.py GRAPH ..."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES, RES_TAB  # noqa: E402
import sys, os, time, numpy as np
sys.path.insert(0, str(GADLIB))
import statistics_cache as C
R, B, F = C.R, C.B, C.F
SEEDS = list(range(101, 116))
def run(name):
    x_raw, a, y = B.load_any(name)
    x, _, _ = R.preprocess(x_raw)
    det = np.concatenate([R.residual_statistics(x, a), F.structural_statistics_fast(x, a)[0]], 1)
    dr, dm, dc, _ = R.exact_marginal(det, a)
    for seed in SEEDS:
        if os.path.exists(f"{C.OUT}/{name}__{seed}.npz"): continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, C.GID[name], 200]))
        loc, _, _ = R.extract(x, a, R.Config(), rng); lr, lm, lc, _ = R.exact_marginal(loc, a)
        C.save(f"{name}__{seed}", np.c_[loc, det], np.c_[lr, dr], np.c_[lm, dm], np.c_[lc, dc])
    print("done", name, flush=True)
for n in sys.argv[1:]:
    t = time.time(); run(n); print(n, f"{time.time()-t:.0f}s", flush=True)
