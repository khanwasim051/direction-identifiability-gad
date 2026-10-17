"""Sensitivity of FDC to the neighbour and pair sampling budgets (q, L) of the contextual statistics (reviewer
comment 8). Recomputes the contextual statistics with (q, L) in {(16, 64), (64, 256)} for seeds 11..71 on a subset
of development graphs and evaluates FDC, FDC-R and fixed directions with the same shots. The default (32, 128) is the
frozen setting. Usage: python fdc/rev_qL.py GRAPH ...  -> results/revision/qL_GRAPH.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES, RES_TAB  # noqa: E402
import sys, os, csv, time, dataclasses, numpy as np
sys.path.insert(0, str(GADLIB)); sys.path.insert(0, str(FDC))
import statistics_cache as C, rev_eval as E
F = E.F; R = C.R; B = C.B
SET = [(16, 64), (32, 128), (64, 256)]
SEEDS = (11, 23, 37, 53, 71)


def main(names):
    for name in names:
        path = f"{E.OUT}/qL_{name}.csv"
        if os.path.exists(path): continue
        t0 = time.time()
        x_raw, a, y = B.load_any(name); y = y.astype(int)
        x, _, _ = R.preprocess(x_raw)
        det = np.concatenate([R.residual_statistics(x, a), C.F.structural_statistics_fast(x, a)[0]], 1)
        dr, dm, dc, _ = R.exact_marginal(det, a)
        ga = F.R.marginal_ranks(E.attr(x_raw, a, 16))
        rows = []
        for q, L in SET:
            cfg = dataclasses.replace(R.Config(), q=q, pair_limit=L)
            for seed in SEEDS:
                rng = np.random.default_rng(np.random.SeedSequence([seed, C.GID[name], 200]))
                loc, _, _ = R.extract(x, a, cfg, rng); lr, lm, lc, _ = R.exact_marginal(loc, a)
                st, rk, mk, ct = (np.c_[loc, det][:, C.COLS], np.c_[lr, dr][:, C.COLS], np.c_[lm, dm][:, C.COLS], np.c_[lc, dc][:, C.COLS])
                if (q, L) == (32, 128):   # must equal the cached default statistics
                    z = np.load(f"{F.CACHE}/{name}__{seed}.npz"); assert np.allclose(z["stats"], st, equal_nan=True)
                fixed = F.score(st, rk, mk, ct, F.FROZEN)
                for k in F.cells(y):
                    sh = F.shots_for(y, seed, k)
                    fdc, v0, _ = E.fdc_scores(st, rk, mk, ct, ga, sh, k)
                    for meth, s in (("FDC", fdc), ("FDC-R", v0), ("Fixed", fixed)):
                        rows.append(dict(dataset=name, q=q, L=L, seed=seed, k=k, method=meth, **E.metrics(y, s, sh)))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
