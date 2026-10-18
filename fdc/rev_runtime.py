"""Revision runtime: one run per graph (seed 11, largest valid k), 2 CPU threads, both methods timed from the loaded
graph in memory (loading from disk timed separately and excluded from both). FDC preprocessing = feature preprocessing,
relational statistics with neighbourhood-excluded ranks, and attribute statistics with their graph-wide ranks
(label-independent, once per graph). FDC calibration = orientation, weighting and scoring with the k labelled anomalies.
GDN = adjacency normalisation, two-step feature propagation, 100 training epochs and scoring of all nodes (official code,
last-epoch variant). Usage: python fdc/rev_runtime.py GRAPH ...  -> results/revision/runtime/runtime_GRAPH.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC, RES  # noqa: E402
import sys, time, csv, os, numpy as np, torch
sys.path.insert(0, str(FDC)); sys.path.insert(0, str(GADLIB))
import fsdc as F, step3 as S3, bench_data as B, fast_stats as FS, statistics_cache as C, run_gdn_official as G, rev_eval as E
R = F.R; torch.set_num_threads(2)
_w = torch.randn(64, 8); G.run(_w, np.r_[np.ones(4), np.zeros(60)].astype(int), np.arange(4), 0)   # warm-up, not timed
for name in sys.argv[1:]:
    out = f"{RES}/revision/runtime/runtime_{name}.csv"; os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): continue
    tl = time.perf_counter(); x_raw, a, y = B.load_any(name); t_load = time.perf_counter() - tl
    y = y.astype(int); k = max(F.cells(y)); sh = F.shots_for(y, 11, k)
    t0 = time.perf_counter()
    x, _, _ = R.preprocess(x_raw)
    det = np.concatenate([R.residual_statistics(x, a), FS.structural_statistics_fast(x, a)[0]], 1)
    dr, dm, dc, _ = R.exact_marginal(det, a)
    rng = np.random.default_rng(np.random.SeedSequence([11, C.GID[name], 200]))
    loc, _, _ = R.extract(x, a, R.Config(), rng); lr, lm, lc, _ = R.exact_marginal(loc, a)
    st, rk, mk, ct = (np.c_[loc, det][:, C.COLS], np.c_[lr, dr][:, C.COLS], np.c_[lm, dm][:, C.COLS], np.c_[lc, dc][:, C.COLS])
    t_rel = time.perf_counter() - t0
    t1 = time.perf_counter(); ga = R.marginal_ranks(E.attr(x_raw, a, 16)); t_attr = time.perf_counter() - t1
    z = np.load(f"{F.CACHE}/{name}__11.npz"); assert np.allclose(z["stats"], st, equal_nan=True)
    t2 = time.perf_counter(); s = S3.final_scores(name, st, rk, mk, ct, ga, sh, k)["W1_kgated"]; t_cal = time.perf_counter() - t2
    t3 = time.perf_counter()
    adj = G.sp_matrix_to_torch_sparse_tensor(G.normalize_adjacency(a.astype(np.float64))).float()
    feat = G.sgc_precompute(torch.FloatTensor(x_raw), adj, 2); G.run(feat, y, sh, 11)
    t_gdn = time.perf_counter() - t3
    row = dict(dataset=name, nodes=len(y), edges=int(a.nnz // 2), k=k, load_s=t_load, fdc_relational_s=t_rel, fdc_attribute_s=t_attr,
               fdc_preprocessing_s=t_rel + t_attr, fdc_calibration_s=t_cal, gdn_official_s=t_gdn)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print(name, {k_: round(v, 3) if isinstance(v, float) else v for k_, v in row.items()}, flush=True)
