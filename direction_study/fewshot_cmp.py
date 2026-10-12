"""Executes FEWSHOT_COMPARISON_PROTOCOL.md exactly."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, glob, os, csv, json, numpy as np
sys.path.insert(0, str(STUDY))
import evaluate as E, rules
from sklearn.metrics import roc_auc_score, average_precision_score
ARC_SETS = ["Cora", "CiteSeer", "ACM", "BlogCatalog", "Facebook", "Weibo-ARC", "Reddit", "Amazon"]
AGFM_SETS = ["Reddit", "Amazon", "Disney", "Amazon-all", "YelpChi-all", "Tolokers", "Questions"]
def excess_tail(rk, mk, shots, rest):
    tails = []
    for j in range(9):
        n = shots[mk[shots, j] > 0]; u = rest[mk[rest, j] > 0]
        if len(n) == 0 or len(u) == 0: tails.append(rules.FROZEN[j]); continue
        up = np.mean(rk[u, j] > rk[n, j].max()); low = np.mean(rk[u, j] < rk[n, j].min())
        tails.append("upper" if up > low else "lower" if low > up else rules.FROZEN[j])
    return tuple(tails)
def anom_rule(rk, mk, lab, yl):
    tails = []
    for j in range(9):
        best, bv = rules.FROZEN[j], -1
        for t in ("upper", "lower", "two-sided"):
            e = rules._evidence(rk[lab, j], mk[lab, j], t)
            v = roc_auc_score(yl, e) if np.ptp(e) > 0 else 0.5
            if v > bv + 1e-12 or (abs(v - bv) <= 1e-12 and t == rules.FROZEN[j]): best, bv = t, v
        tails.append(best)
    return tuple(tails)
def main():
    rows = []
    for name in sorted(set(ARC_SETS + AGFM_SETS)):
        y = np.load(f"{E.C}/labels_{name}.npy")
        for seed in (11, 23, 37, 53, 71):
            z = np.load(f"{E.C}/{name}__{seed}.npz"); st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
            def score(t): return E.R.score_bank(st, rk, mk, ct, tuple(range(6)), (6, 7, 8), t)
            for k in (1, 5, 10):
                g = np.random.default_rng(np.random.SeedSequence([seed, 7000 + k]))
                shots = g.choice(np.flatnonzero(y == 0), k, replace=False)
                rest = np.setdiff1d(np.arange(len(y)), shots)
                for meth, t in (("normal_shot_excess_tail", excess_tail(rk, mk, shots, rest)), ("frozen_zero_shot", rules.FROZEN)):
                    s = score(t)
                    rows.append(dict(dataset=name, seed=seed, k=k, method=meth, auroc=roc_auc_score(y[rest], s[rest]), ap=average_precision_score(y[rest], s[rest]), tails="".join(x[0] for x in t)))
            if y.sum() >= 11:   # secondary: 10 anomalies + 10 normals
                g = np.random.default_rng(np.random.SeedSequence([seed, 9000 + 10]))
                pa = g.choice(np.flatnonzero(y == 1), 10, replace=False); pn = g.choice(np.flatnonzero(y == 0), 10, replace=False)
                lab = np.r_[pa, pn]; rest = np.setdiff1d(np.arange(len(y)), lab)
                t = anom_rule(rk, mk, lab, np.r_[np.ones(10), np.zeros(10)]); s = score(t)
                rows.append(dict(dataset=name, seed=seed, k=10, method="anomaly+normal_shots", auroc=roc_auc_score(y[rest], s[rest]), ap=average_precision_score(y[rest], s[rest]), tails="".join(x[0] for x in t)))
        print("done", name, flush=True)
    with open(RES_STUDY / "results_fewshot_cmp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
