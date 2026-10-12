import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, time, numpy as np
sys.path.insert(0, str(BENCH))
PKG = str(PKG)
sys.path.insert(0, PKG + "/core_multiscale_directional"); sys.path.insert(0, PKG + "/baseline_comparison")
import run_experiment as R, bench_data as B, fast_stats as F
from run_neighbor_div import projected_features, neighbor_div_scores
from scipy.stats import spearmanr
f0,a0,_ = R.citation_instance("cora", "cross_community_edges", 53)
cases = [("cora-ccedge", f0, a0, f0)]
for name in ["Facebook", "Disney", "BlogCatalog", "Reddit"]:
    raw, a, y = B.load(name); x, _, _ = R.preprocess(raw); cases.append((name, x, a, raw))
for name, x, a, raw in cases:
    t = time.perf_counter(); s0, tc0 = R.structural_statistics(x, a); t0 = time.perf_counter() - t
    t = time.perf_counter(); s1, tc1 = F.structural_statistics_fast(x, a); t1 = time.perf_counter() - t
    nanmatch = np.array_equal(np.isnan(s0), np.isnan(s1))
    diff = np.nanmax(np.abs(s0 - s1), axis=0)
    print(f"{name}: struct tri_equal={np.array_equal(tc0, tc1)} nan_equal={nanmatch} maxdiff={diff} t_orig={t0:.1f}s t_fast={t1:.1f}s", flush=True)
    if name in ("Facebook", "Disney", "BlogCatalog", "cora-ccedge"):
        emb = projected_features(raw)
        t = time.perf_counter(); n0 = neighbor_div_scores(emb, a); u0 = time.perf_counter() - t
        t = time.perf_counter(); n1 = F.neighbor_div_fast(emb, a); u1 = time.perf_counter() - t
        print(f"   neighbordiv maxdiff_div={np.nanmax(np.abs(n0[1]-n1[1])):.2e} maxdiff_score={np.max(np.abs(n0[0]-n1[0])):.2e} t_orig={u0:.1f}s t_fast={u1:.1f}s", flush=True)
