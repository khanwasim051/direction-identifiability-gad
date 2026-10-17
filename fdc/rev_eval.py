"""Post hoc revision analyses (all on the frozen method; nothing here changes FDC):
 * 20 seeds (11,23,37,53,71 and 101-115) for FDC, FDC-R and lightweight baselines;
 * extra metrics: normalised AP and R-precision (precision at K = number of evaluated anomalies);
 * lightweight baselines that use the same k labelled anomalies: shot similarity (max / centroid cosine in the
   attribute space), positive-unlabelled logistic regression, and the single best calibrated statistic;
 * sensitivity of FDC to the threshold multiplier c (tau = c / sqrt(12k)), the k-gate and the attribute dimension p;
 * null experiment: k uniformly random nodes as 'shots'; number of statistics retained and resulting AUROC.
Usage: python fdc/rev_eval.py GRAPH ...  -> results/revision/eval_GRAPH.csv, results/revision/null_GRAPH.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES, RES_TAB  # noqa: E402
import sys, os, csv, time, numpy as np, scipy.sparse as sp
sys.path.insert(0, str(FDC)); sys.path.insert(0, str(GADLIB))
import fsdc as F, step2 as S2, bench_data as B
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = str(RES / "revision"); os.makedirs(OUT, exist_ok=True)
SEALED = str(SEALED); HELD = {"CS", "Photo", "Books", "Enron"}
SEEDS = (11, 23, 37, 53, 71) + tuple(range(101, 116))


def attr(x, a, p):
    n, d = x.shape; p = min(p, d - 1)
    if d > 64 or (x != 0).mean() < 0.3:
        xr = x / np.maximum(np.abs(x).sum(1, keepdims=True), 1e-12)
        Z = TruncatedSVD(p, random_state=0).fit_transform(sp.csr_matrix(xr))
    else:
        xs = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
        Z = PCA(p, random_state=0).fit_transform(xs)
    a = sp.csr_matrix(a, dtype=np.float64); deg = np.asarray(a.sum(1)).ravel()
    nb = (a @ Z) / np.maximum(deg, 1)[:, None]
    return np.c_[Z, nb]


def metrics(y, s, shots):
    rest = np.setdiff1d(np.arange(len(y)), shots); yr, sr = y[rest], s[rest]
    au, ap = roc_auc_score(yr, sr), average_precision_score(yr, sr)
    pi = yr.mean(); K = int(yr.sum())
    top = np.argsort(-sr, kind="stable")[:K]
    return dict(auroc=au, ap=ap, nap=(ap - pi) / (1 - pi), rprec=yr[top].mean())


def fdc_scores(st, rk, mk, ct, ga, sh, k, c=1.0, gate=3):
    """FDC with threshold multiplier c and k-gate; c=1, gate=3 is the frozen method (identical code path)."""
    grel = F.R.marginal_ranks(st)
    t9, s9 = F.calibrate(F.shot_aucs(grel, mk, sh))
    v0 = F.score(st, rk, mk, ct, t9, s9)
    if k < gate: return v0, v0, None
    gr = np.c_[grel, ga]; m = np.c_[mk, np.ones_like(ga)]
    frozen = list(F.FROZEN) + ["upper"] * ga.shape[1]
    tails, s = S2.calibrate(gr, m, sh, frozen)
    w = np.maximum(0, s - c / np.sqrt(12 * k))
    if w.sum() == 0: w = s
    return S2.pooled(gr, m, tails, w), v0, (tails, s, w)


def single_best(gr, m, sh, frozen):
    tails, s = S2.calibrate(gr, m, sh, frozen)
    j = int(np.argmax(s)); w = np.zeros_like(s); w[j] = 1.0
    return S2.pooled(gr, m, tails, w)


def main(names):
    for name in names:
        path = f"{OUT}/eval_{name}.csv"
        if os.path.exists(path): continue
        t0 = time.time()
        x, a, y = B.load_any(name); y = y.astype(int)
        lab = np.load(f"{SEALED if name in HELD else F.CACHE}/labels_{name}.npy").astype(int); assert (lab == y).all()
        G = {p: attr(x, a, p) for p in (8, 16, 32)}
        GR = {p: F.R.marginal_ranks(G[p]) for p in G}
        Zs = (G[16] - G[16].mean(0)) / np.where(G[16].std(0) > 0, G[16].std(0), 1)
        Zn = Zs / np.maximum(np.linalg.norm(Zs, axis=1, keepdims=True), 1e-12)
        rows, nrows = [], []
        for seed in SEEDS:
            z = np.load(f"{F.CACHE}/{name}__{seed}.npz"); st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
            grel = F.R.marginal_ranks(st)
            for k in F.cells(y):
                sh = F.shots_for(y, seed, k)
                def rec(meth, s): rows.append(dict(dataset=name, seed=seed, k=k, method=meth, **metrics(y, s, sh)))
                fdc, v0, _ = fdc_scores(st, rk, mk, ct, GR[16], sh, k)
                rec("FDC", fdc); rec("FDC-R", v0)
                for c in (0.0, 0.5, 1.5, 2.0):
                    rec(f"FDC_c{c}", fdc_scores(st, rk, mk, ct, GR[16], sh, k, c=c)[0])
                for gate in (1, 4):   # with k in {1,3,5,10}: gate 2 = gate 3 (FDC), gate 5 = gate 4
                    rec(f"FDC_gate{gate}", fdc_scores(st, rk, mk, ct, GR[16], sh, k, gate=gate)[0])
                for p in (8, 32):
                    rec(f"FDC_p{p}", fdc_scores(st, rk, mk, ct, GR[p], sh, k)[0])
                # lightweight baselines with the same k anomalies
                sim = Zn @ Zn[sh].T
                rec("ShotSim-max", sim.max(1)); rec("ShotSim-centroid", Zn @ Zn[sh].mean(0))
                lab_pu = np.zeros(len(y), int); lab_pu[sh] = 1
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Zs, lab_pu)
                rec("PU-logistic", lr.decision_function(Zs))
                gr = np.c_[grel, GR[16]]; m = np.c_[mk, np.ones_like(GR[16])]
                rec("BestSingleStat", single_best(gr, m, sh, list(F.FROZEN) + ["upper"] * GR[16].shape[1]))
                # null experiment: k random nodes as shots
                g = np.random.default_rng(np.random.SeedSequence([seed, 13000 + k]))
                rnd = np.sort(g.choice(len(y), k, replace=False))
                s_null, _, info_null = fdc_scores(st, rk, mk, ct, GR[16], rnd, k, gate=1)
                _, _, info_real = fdc_scores(st, rk, mk, ct, GR[16], sh, k, gate=1)
                nrows.append(dict(dataset=name, seed=seed, k=k, n_stats=len(info_null[2]),
                                  kept_null=int((info_null[2] > 0).sum()), kept_real=int((info_real[2] > 0).sum()),
                                  auroc_null=metrics(y, s_null, rnd)["auroc"]))
        for p_, r in ((path, rows), (f"{OUT}/null_{name}.csv", nrows)):
            with open(p_, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(r[0])); w.writeheader(); w.writerows(r)
        print(name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
