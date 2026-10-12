import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, glob, os, csv, json, numpy as np
sys.path.insert(0, str(STUDY))
import evaluate as E, rules
from sklearn.metrics import roc_auc_score, average_precision_score
rows = []
for split, units in E.SPLITS.items():
    for unit, pat in units.items():
        for f in sorted(glob.glob(f"{E.C}/{pat}.npz")):
            z = np.load(f); y = E.labels_for(f, split); st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
            seed = int(f.split("__")[1][:-4])
            for k in (5, 10):
                if y.sum() < k + 1: continue
                g = np.random.default_rng(np.random.SeedSequence([seed, 9000 + k]))
                pa = g.choice(np.flatnonzero(y == 1), k, replace=False); pn = g.choice(np.flatnonzero(y == 0), k, replace=False)
                lab = np.r_[pa, pn]; yl = np.r_[np.ones(k), np.zeros(k)]
                tails = []
                for j in range(9):
                    best, bv = rules.FROZEN[j], -1
                    for t in ("upper", "lower", "two-sided"):
                        e = rules._evidence(rk[lab, j], mk[lab, j], t)
                        v = roc_auc_score(yl, e) if np.ptp(e) > 0 else 0.5
                        if v > bv + 1e-12 or (abs(v - bv) <= 1e-12 and t == rules.FROZEN[j]): best, bv = t, v
                    tails.append(best)
                rest = np.ones(len(y), bool); rest[lab] = False
                for name, tl in (("fewshot", tuple(tails)), ("frozen", rules.FROZEN)):
                    s = E.R.score_bank(st, rk, mk, ct, tuple(range(6)), (6, 7, 8), tl)
                    rows.append(dict(split=split, unit=unit, run=os.path.basename(f)[:-4], k=k, method=name,
                                     auroc=roc_auc_score(y[rest], s[rest]), ap=average_precision_score(y[rest], s[rest])))
with open(RES_STUDY / "results_fewshot.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
for k in (5, 10):
    print(f"== k={k}")
    for split in ("dev", "t1", "t2"):
        us = sorted({r["unit"] for r in rows if r["split"] == split and r["k"] == k})
        m = {meth: [np.mean([r["auroc"] for r in rows if r["unit"] == u and r["k"] == k and r["method"] == meth]) for u in us] for meth in ("frozen", "fewshot")}
        a = {meth: [np.mean([r["ap"] for r in rows if r["unit"] == u and r["k"] == k and r["method"] == meth]) for u in us] for meth in ("frozen", "fewshot")}
        print(f" {split}: frozen AUROC={np.mean(m['frozen']):.3f} AP={np.mean(a['frozen']):.3f} | fewshot AUROC={np.mean(m['fewshot']):.3f} AP={np.mean(a['fewshot']):.3f} | wins {sum(x>y for x,y in zip(m['fewshot'],m['frozen']))}/{len(us)}")
        print("   " + " ".join(f"{u[:10]}:{f0:.2f}->{f1:.2f}" for u, f0, f1 in zip(us, m["frozen"], m["fewshot"])))
