"""Cache frozen-pipeline statistics/ranks/masks/counts per graph and seed. Labels saved separately;
T2 labels go to a sealed directory that is not read before the rule is frozen."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, os, time, numpy as np
sys.path.insert(0, str(BENCH))
PKG = str(PKG); sys.path.insert(0, PKG + "/core_multiscale_directional")
import run_experiment as R, bench_data as B, fast_stats as F
OUT = str(WORK / "cache"); SEALED = str(WORK / "sealed_labels")
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
def synthetic():
    specs = [(g, s, "dev") for g in ("cora", "citeseer") for s in (11, 23, 37)] + \
            [(g, s, "hold") for g in ("cora", "citeseer") for s in (53, 71)] + [("pubmed", s, "transfer") for s in (11, 23, 37, 53, 71)]
    for g, s, ph in specs:
        for mech in R.MECHANISMS:
            tag = f"syn_{ph}_{g}_{mech}__{s}"
            if os.path.exists(f"{OUT}/{tag}.npz"): continue
            f, a, stage = R.citation_instance(g, mech, s)
            st, rk, mk, ct, _ = R.construct(f, a, s, R.GRAPH_IDS[g], stage)
            save(tag, st, rk, mk, ct); np.save(f"{OUT}/labels_{tag}.npy", R.citation_labels(g, mech, s))
    print("done synthetic", flush=True)
if __name__ == "__main__":
    for n in sys.argv[1:]:
        t = time.perf_counter(); synthetic() if n == "synthetic" else standard(n); print(n, f"{time.perf_counter()-t:.0f}s", flush=True)
