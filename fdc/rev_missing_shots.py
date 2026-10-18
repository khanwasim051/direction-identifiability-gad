"""Revision (threshold with missing values): how often |S_j| < k for the k labelled anomalies, effect on retention, all-zero fallback, and a post hoc variant
with tau based on |S_j| (not the frozen method).
Usage: python fdc/rev_missing_shots.py  -> results/revision/sj_counts.csv, sj_variant.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC  # noqa: E402
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, str(FDC)); sys.path.insert(0, str(GADLIB))
import fsdc as F, step2 as S2, rev_eval as E, bench_data as B
HELD = {"CS", "Photo", "Books", "Enron"}
names = sorted(pd.read_csv(p).dataset.iloc[0] for p in glob.glob(f"{E.OUT}/eval_*.csv"))
rows, cnt = [], []
for name in names:
    x, a, y = B.load_any(name); y = y.astype(int)
    GR = F.R.marginal_ranks(E.attr(x, a, 16))
    for seed in E.SEEDS:
        z = np.load(f"{F.CACHE}/{name}__{seed}.npz"); st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
        grel = F.R.marginal_ranks(st)
        for k in F.cells(y):
            sh = F.shots_for(y, seed, k)
            nsj = mk[sh].sum(0)                             # |S_j| for the 9 relational statistics
            fdc, v0, info = E.fdc_scores(st, rk, mk, ct, GR, sh, k)
            if k < 3: continue
            tails, s, w = info
            gr = np.c_[grel, GR]; m = np.c_[mk, np.ones_like(GR)]
            nall = np.r_[nsj, np.full(GR.shape[1], k)]
            tau_j = 1 / np.sqrt(12 * np.maximum(nall, 1))
            w2 = np.maximum(0, s - tau_j); w2[nall == 0] = 0
            allzero_w = bool(w.sum() == 0) if False else bool(np.maximum(0, s - 1 / np.sqrt(12 * k)).sum() == 0)
            if w2.sum() == 0: w2 = s
            s2 = S2.pooled(gr, m, tails, w2) if w2.sum() > 0 else np.zeros(len(y))
            for st_j in range(9):
                cnt.append(dict(dataset=name, seed=seed, k=k, j=st_j, nsj=int(nsj[st_j]), short=bool(nsj[st_j] < k),
                                kept=bool(np.maximum(0, s[st_j] - 1 / np.sqrt(12 * k)) > 0), kept_sj=bool(w2[st_j] > 0) if nsj[st_j] > 0 else False))
            rows.append(dict(dataset=name, seed=seed, k=k, split="held-out" if name in HELD else "development", allzero=allzero_w,
                             allzero_s=bool(s.sum() == 0), **{f"{kk}_frozen": vv for kk, vv in E.metrics(y, fdc, sh).items()},
                             **{f"{kk}_sj": vv for kk, vv in E.metrics(y, s2, sh).items()}))
    print(name, flush=True)
pd.DataFrame(cnt).to_csv(f"{E.OUT}/sj_counts.csv", index=False); pd.DataFrame(rows).to_csv(f"{E.OUT}/sj_variant.csv", index=False)
