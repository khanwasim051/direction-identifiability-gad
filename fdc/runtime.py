"""Wall-clock runtime of the full FINAL pipeline (statistics from raw graph + calibration + scoring) and of official
GDN, one run each (seed 11, k = 10 or the largest valid k), same machine, 2 CPU threads. Output: results/tables/runtime_<graph>.csv and the combined results/tables/runtime.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import PKG, GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES_R1, RES_DEV, RES_R23, RES_HO, RES_TAB  # noqa: E402
import sys, time, csv, numpy as np, torch
sys.path.insert(0, str(FDC)); sys.path.insert(0, str(GADLIB))
import fsdc as F, step2 as S2, step3 as S3, bench_data as B, fast_stats as FS, statistics_cache as C, run_gdn_official as G
R = F.R; torch.set_num_threads(2)
rows = []
for name in sys.argv[1:]:
    x_raw, a, y = B.load_any(name); y = y.astype(int); k = max(F.cells(y)); sh = F.shots_for(y, 11, k)
    t0 = time.perf_counter()
    x, _, _ = R.preprocess(x_raw)
    det = np.concatenate([R.residual_statistics(x, a), FS.structural_statistics_fast(x, a)[0]], 1)
    dr, dm, dc, _ = R.exact_marginal(det, a)
    rng = np.random.default_rng(np.random.SeedSequence([11, C.GID[name], 200]))
    loc, _, _ = R.extract(x, a, R.Config(), rng); lr, lm, lc, _ = R.exact_marginal(loc, a)
    st, rk, mk, ct = (np.c_[loc, det][:, C.COLS], np.c_[lr, dr][:, C.COLS], np.c_[lm, dm][:, C.COLS], np.c_[lc, dc][:, C.COLS])
    t_stats = time.perf_counter() - t0
    t1 = time.perf_counter(); ga, _ = S2.attribute_stats(name); t_attr = time.perf_counter() - t1
    t2 = time.perf_counter(); s = S3.final_scores(name, st, rk, mk, ct, ga, sh, k)["W1_kgated"]; t_cal = time.perf_counter() - t2
    import scipy.sparse as sp
    t3 = time.perf_counter()
    adj = G.sp_matrix_to_torch_sparse_tensor(G.normalize_adjacency(a.astype(np.float64))).float()
    feat = G.sgc_precompute(torch.FloatTensor(x_raw), adj, 2); G.run(feat, y, sh, 11)
    t_gdn = time.perf_counter() - t3
    rows.append(dict(dataset=name, nodes=len(y), edges=int(a.nnz // 2), k=k, final_statistics_s=round(t_stats, 2),
                     final_attribute_s=round(t_attr, 2), final_calibration_s=round(t_cal, 3),
                     final_total_s=round(t_stats + t_attr + t_cal, 2), gdn_official_s=round(t_gdn, 2)))
    print(rows[-1], flush=True)
    with open(f"{RES_TAB}/runtime_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1])); w.writeheader(); w.writerow(rows[-1])
import glob, pandas as pd
d = pd.concat([pd.read_csv(p) for p in glob.glob(f"{RES_TAB}/runtime_*.csv")]).sort_values("edges")
d["speedup_vs_gdn"] = (d.gdn_official_s / d.final_total_s).round(1); d.to_csv(f"{RES_TAB}/runtime.csv", index=False)
