"""NeighborDiv and FreeGAD (user's equation-faithful implementations) on the benchmark files.
NeighborDiv uses the exact closed-form pair variance (verified equal to the exhaustive loop);
FreeGAD uses a two-pass, memory-bounded version of the same equations (verified below)."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, os, csv, time, gc
import numpy as np
sys.path.insert(0, str(BENCH))
PKG = str(PKG)
sys.path.insert(0, PKG + "/baseline_comparison")
import bench_data as B, fast_stats as F
import run_freegad as FG
from run_neighbor_div import projected_features
from sklearn.metrics import roc_auc_score, average_precision_score

def freegad_embedding_lowmem(features, adjacency):
    P = FG.symmetric_normalize_with_self_loops(adjacency)
    base_norm = np.linalg.norm(features, axis=1) + 1e-10
    cur = features; cos = []
    for _ in range(FG.NUM_HOPS):
        cur = np.asarray(P @ cur, dtype=np.float32)
        cos.append(np.einsum("ij,ij->i", cur, features) / ((np.linalg.norm(cur, axis=1) + 1e-10) * base_norm))
    logits = np.column_stack(cos).astype(np.float32); logits -= logits.max(1, keepdims=True)
    w = np.exp(logits); w /= w.sum(1, keepdims=True)
    emb = np.zeros_like(features, dtype=np.float32); cur = features
    for h in range(FG.NUM_HOPS):
        cur = np.asarray(P @ cur, dtype=np.float32)
        emb += (1.0 - w[:, h:h+1]) * cur + w[:, h:h+1] * features
    return emb / FG.NUM_HOPS

def freegad_scores_lowmem(features, adjacency):
    orig = FG.affinity_gated_embedding
    FG.affinity_gated_embedding = freegad_embedding_lowmem
    try:
        return FG.freegad_scores(features, adjacency)
    finally:
        FG.affinity_gated_embedding = orig

def run(name):
    import scipy.sparse as sp
    x, a, y = B.load(name); rows = []
    t = time.perf_counter(); s = F.neighbor_div_fast(projected_features(sp.csr_matrix(x, dtype=np.float64) if x.shape[1] > 1000 else x), a)[0]; el = time.perf_counter() - t
    rows.append(dict(method="NeighborDiv", dataset=name, auroc=roc_auc_score(y, s), ap=average_precision_score(y, s), seconds=el))
    t = time.perf_counter(); xn = FG.row_normalize_features(x); del x; gc.collect()
    s = freegad_scores_lowmem(xn, a)[0]; el = time.perf_counter() - t
    rows.append(dict(method="FreeGAD", dataset=name, auroc=roc_auc_score(y, s), ap=average_precision_score(y, s), seconds=el))
    for r in rows: print(f"{name} {r['method']} AUROC={r['auroc']:.4f} AP={r['ap']:.4f} t={r['seconds']:.1f}s", flush=True)
    return rows

if __name__ == "__main__":
    if sys.argv[1] == "verify":
        for name in ("Facebook", "Disney", "Cora", "Reddit"):
            x, a, y = B.load(name); xn = FG.row_normalize_features(x)
            s0 = FG.freegad_scores(xn, a)[0]; s1 = freegad_scores_lowmem(xn, a)[0]
            print(name, "FreeGAD max|diff|", np.abs(s0 - s1).max(), "AUROC", roc_auc_score(y, s0), roc_auc_score(y, s1), flush=True)
        sys.exit()
    out = str(RES_BENCH / "tf_out"); os.makedirs(out, exist_ok=True)
    for name in sys.argv[1:]:
        rows = run(name)
        with open(f"{out}/metrics_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        gc.collect()
