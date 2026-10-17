"""Executes STEP2_PROTOCOL.md (sha256 in STEP2_PROTOCOL.sha256). Output: results3/step2.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import PKG, GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES_R1, RES_DEV, RES_R23, RES_HO, RES_TAB  # noqa: E402
import sys, os, csv, numpy as np, scipy.sparse as sp
sys.path.insert(0, str(FDC)); sys.path.insert(0, str(GADLIB))
import fsdc as F, bench_data as B
from sklearn.decomposition import PCA, TruncatedSVD

OUT = str(RES_R23); os.makedirs(OUT, exist_ok=True)
INJECTED = {"Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook"}


def attribute_stats(name):
    x, a, y = B.load_any(name)
    n, d = x.shape; p = min(16, d - 1)
    dens = (x != 0).mean()
    if d > 64 or dens < 0.3:
        xr = x / np.maximum(np.abs(x).sum(1, keepdims=True), 1e-12)
        Z = TruncatedSVD(p, random_state=0).fit_transform(sp.csr_matrix(xr))
    else:
        xs = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
        Z = PCA(p, random_state=0).fit_transform(xs)
    a = sp.csr_matrix(a, dtype=np.float64); deg = np.asarray(a.sum(1)).ravel()
    nb = (a @ Z) / np.maximum(deg, 1)[:, None]
    S = np.c_[Z, nb]
    return F.R.marginal_ranks(S), y.astype(int)


def oriented(r, tail):
    return (r + 1) / 2 if tail == "upper" else (1 - r) / 2 if tail == "lower" else np.abs(r)


def calibrate(gr, mk, shots, frozen):
    """Per statistic: tail and s_j from the anomaly shots (same rule as M2)."""
    tails, sep = [], []
    for j in range(gr.shape[1]):
        a = shots[mk[shots, j] > 0]
        if len(a) == 0: tails.append(frozen[j]); sep.append(0.0); continue
        up = float(np.mean((gr[a, j] + 1) / 2))
        d = {"upper": up, "lower": 1 - up, "two-sided": float(np.mean(np.abs(gr[a, j])))}
        best = max(d.values()); c = [t for t in ("upper", "lower", "two-sided") if abs(d[t] - best) <= 1e-12]
        t = frozen[j] if frozen[j] in c else c[0]
        tails.append(t); sep.append(max(0.0, d[t] - 0.5))
    return tails, np.array(sep)


def pooled(gr, mk, tails, w):
    E = np.column_stack([oriented(gr[:, j], tails[j]) for j in range(gr.shape[1])]) * mk
    den = (mk * w[None, :]).sum(1)
    return np.divide((E * w[None, :]).sum(1), den, out=np.zeros(len(gr)), where=den > 0)


def main():
    rows = []
    for name in F.GRAPHS:
        ga, y = attribute_stats(name)
        assert (np.load(f"{F.CACHE}/labels_{name}.npy").astype(int) == y).all()
        for seed in F.SEEDS:
            z = np.load(f"{F.CACHE}/{name}__{seed}.npz"); st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
            grel = F.R.marginal_ranks(st)
            gr = np.c_[grel, ga]; m = np.c_[mk, np.ones_like(ga)]
            frozen = list(F.FROZEN) + ["upper"] * ga.shape[1]
            for k in F.cells(y):
                sh = F.shots_for(y, seed, k)
                # V0 = M2 exactly as in fsdc.py
                t9, s9 = F.calibrate(F.shot_aucs(grel, mk, sh))
                v0 = F.score(st, rk, mk, ct, t9, s9)
                tails, s = calibrate(gr, m, sh, frozen)
                v1 = pooled(gr, m, tails, s)
                tau = 1 / np.sqrt(12 * k); w2 = np.maximum(0, s - tau)
                if w2.sum() == 0: w2 = s
                v2 = pooled(gr, m, tails, w2)
                w3 = w2.copy(); keep = np.argsort(-s, kind="stable")[:8]; mask = np.zeros_like(w3); mask[keep] = 1; w3 = w3 * mask
                if w3.sum() == 0: w3 = w2
                v3 = pooled(gr, m, tails, w3)
                for meth, sc in (("V0_M2", v0), ("V1_pooled", v1), ("V2_shrink", v2), ("V3_top8", v3)):
                    au, ap = F.metrics(y, sc, sh)
                    rows.append(dict(dataset=name, seed=seed, k=k, method=meth, auroc=au, ap=ap,
                                     group="injected" if name in INJECTED else "organic"))
        print("done", name, flush=True)
    with open(f"{OUT}/step2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
