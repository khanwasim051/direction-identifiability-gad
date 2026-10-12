"""Every number, table and figure in the DMKD paper is generated here from saved results."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, glob, os, json, csv, numpy as np
sys.path.insert(0, str(STUDY)); sys.path.insert(0, str(BENCH))
import evaluate as E, rules
from sklearn.metrics import roc_auc_score, average_precision_score
C = E.C; OUT = str(RES_PAPER)
SEEDS = (11, 23, 37, 53, 71)
INJ = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook"]
ZS = ["Reddit", "Amazon", "Disney", "Weibo", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi"]
T2 = ["CS", "Photo", "Books", "Enron"]
ALLG = INJ + ZS + ["Weibo-ARC"] + T2
def labels(n): return np.load(f"{E.SEALED}/labels_{n}.npy") if n in T2 else np.load(f"{C}/labels_{n}.npy")
def load(n, s): z = np.load(f"{C}/{n}__{s}.npz"); return z["stats"], z["ranks"], z["mask"], z["counts"]
def score(st, rk, mk, ct, tails, ctx=tuple(range(6)), strc=(6, 7, 8)): return E.R.score_bank(st, rk, mk, ct, ctx, strc, tails)
N = {}
# ---------- frozen detector + fixed variants on all graphs (cache-consistent) ----------
VAR = {"frozen": (rules.FROZEN, tuple(range(6)), (6, 7, 8)), "twosided": (("two-sided",) * 9, tuple(range(6)), (6, 7, 8)),
       "no_triangle": (rules.FROZEN, tuple(range(6)), (7,))}
core = {}
for g in ALLG:
    y = labels(g); core[g] = {}
    for v, (t, cx, sx) in VAR.items():
        au, ap = [], []
        for s in SEEDS:
            st, rk, mk, ct = load(g, s); sc = score(st, rk, mk, ct, t, cx, sx)
            au.append(roc_auc_score(y, sc)); ap.append(average_precision_score(y, sc))
        core[g][v] = dict(auroc=float(np.mean(au)), auroc_sd=float(np.std(au, ddof=1)), ap=float(np.mean(ap)), ap_sd=float(np.std(ap, ddof=1)))
    core[g]["n"] = int(len(y)); core[g]["anom"] = int(y.sum())
N["core"] = core
# graph statistics
import bench_data as B
gstat = {}
for g in ALLG:
    st = np.load(f"{C}/{g}__11.npz")["stats"]; y = labels(g)
    x, a, _ = B.load_any(g)
    gstat[g] = dict(nodes=int(a.shape[0]), edges=int(a.nnz // 2), features=int(x.shape[1]), anomalies=int(y.sum()), rate=float(y.mean()))
N["graphs"] = gstat
# ---------- baselines ----------
ar = json.load(open(RES_BENCH / "all_results.json")); N["baselines"] = ar
# ---------- direction diagnostic: AUROC of each oriented coordinate, mean over seeds ----------
diag = {}
for g in ALLG:
    y = labels(g); vals = []
    for s in SEEDS:
        st, rk, mk, ct = load(g, s); row = []
        for j in range(9):
            v = mk[:, j] > 0
            row.append(roc_auc_score(y[v], rules._evidence(rk[v, j], 1.0, rules.FROZEN[j])) if v.sum() > 10 and 0 < y[v].sum() < v.sum() else np.nan)
        vals.append(row)
    diag[g] = [None if np.isnan(x) else float(x) for x in np.nanmean(np.array(vals, float), 0)] if not np.all(np.isnan(vals)) else None
N["diag"] = diag
# ---------- rule study ----------
N["rules"] = {sp: json.load(open(f"{RES_STUDY}/summary_{sp}.json")) for sp in ("dev", "t1", "t2")}
N["freeze"] = json.load(open(RES_STUDY / "freeze.json"))
# ---------- few-shot comparison ----------
R = list(csv.DictReader(open(RES_STUDY / "results_fewshot_cmp.csv")))
fs = {}
for r in R:
    key = f"{r['method']}|{r['k']}|{r['dataset']}"; fs.setdefault(key, []).append((float(r["auroc"]), float(r["ap"])))
N["fewshot"] = {k: dict(auroc=float(np.mean([a for a, _ in v])), ap=float(np.mean([b for _, b in v])), runs=len(v)) for k, v in fs.items()}
# ---------- direction recovery analysis (post hoc) ----------
# oracle direction of a decisive statistic: upper vs lower by AUROC over all labeled nodes, decisive if |AUROC_upper-0.5| >= 0.1
def oracle(rk, mk, y):
    out = {}
    for j in range(9):
        v = mk[:, j] > 0
        if v.sum() < 10 or not (0 < y[v].sum() < v.sum()): continue
        au = roc_auc_score(y[v], rk[v, j])
        if abs(au - 0.5) >= 0.1: out[j] = "upper" if au > 0.5 else "lower"
    return out
sources = {}
def add(src, g, tails, orc):
    for j, d in orc.items():
        sources.setdefault(src, {}).setdefault(g, []).append(tails[j] == d)
import fewshot_cmp as FC
for g in ALLG:
    y = labels(g)
    for s in SEEDS:
        st, rk, mk, ct = load(g, s); orc = oracle(rk, mk, y)
        if not orc: continue
        add("Fixed (frozen) directions", g, rules.FROZEN, orc)
        add("Skewness rule A(kelly,0.1)", g, rules.rule_A(st, "kelly", 0.1), orc)
        add("Consensus rule C", g, rules.rule_consensus(rk, mk, rules.FROZEN), orc)
        for k in (1, 5, 10):
            gg = np.random.default_rng(np.random.SeedSequence([s, 7000 + k])); shots = gg.choice(np.flatnonzero(y == 0), k, replace=False)
            rest = np.setdiff1d(np.arange(len(y)), shots)
            add(f"{k} normal shots", g, FC.excess_tail(rk, mk, shots, rest), orc)
        for k in (1, 3, 5, 10):
            if y.sum() < k + 1: continue
            gg = np.random.default_rng(np.random.SeedSequence([s, 9100 + k]))
            pa = gg.choice(np.flatnonzero(y == 1), k, replace=False); pn = gg.choice(np.flatnonzero(y == 0), k, replace=False)
            add(f"{k} anomaly + {k} normal shots", g, FC.anom_rule(rk, mk, np.r_[pa, pn], np.r_[np.ones(k), np.zeros(k)]), orc)
rec = {src: dict(graph_mean=float(np.mean([np.mean(v) for v in d.values()])), pooled=float(np.mean([x for v in d.values() for x in v])), graphs=len(d), decisions=sum(len(v) for v in d.values()))
       for src, d in sources.items()}
N["recovery"] = rec
json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1)
for k, v in rec.items(): print(f"{k:32s} {v}")
