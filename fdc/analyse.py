"""Applies the go/no-go criteria of FEASIBILITY_PROTOCOL.md and the secondary bound check."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import PKG, GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES_R1, RES_DEV, RES_R23, RES_HO, RES_TAB  # noqa: E402
import glob, json, sys, numpy as np, pandas as pd
sys.path.insert(0, str(FDC))
import fsdc as F

ours = pd.read_csv(f"{F.OUT}/ours.csv")
base = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{F.OUT}/base_*.csv"))])
d = pd.concat([ours[["dataset", "seed", "k", "method", "auroc", "ap"]], base])
missing = set(F.GRAPHS) - set(base.dataset)
m = d.groupby(["dataset", "k", "method"])[["auroc", "ap"]].mean().reset_index()
A = m.pivot_table(index=["k", "dataset"], columns="method", values="auroc")
P = m.pivot_table(index=["k", "dataset"], columns="method", values="ap")
A["best_base"] = A[["B1_GDN", "B2_BWGNN"]].max(1)

out = {"missing_baseline_graphs": sorted(missing)}
for k in F.KS:
    a = A.loc[k]
    out[f"k={k}"] = dict(n_graphs=len(a), **{c: round(a[c].mean(), 4) for c in a.columns},
                         M2_wins_vs_best_base=int((a["M2_DIR-W"] > a["best_base"]).sum()),
                         AP={c: round(P.loc[k][c].mean(), 4) for c in P.columns})
c1 = all(out[f"k={k}"]["M2_DIR-W"] >= out[f"k={k}"]["best_base"] - 0.01 for k in (5, 10))
c2 = out["k=10"]["M2_wins_vs_best_base"] >= out["k=10"]["n_graphs"] / 2
c3 = all(out[f"k={k}"]["M2_DIR-W"] > out[f"k={k}"]["M0_frozen"] for k in (5, 10))
out["criteria"] = dict(C1_mean_within_001=c1, C2_wins_half_k10=c2, C3_beats_frozen=c3, GO=bool(c1 and c2 and c3) and not missing)

# secondary: up/low direction error vs Hoeffding bound exp(-2 k delta^2)
rows = []
for name in F.GRAPHS:
    y = np.load(f"{F.CACHE}/labels_{name}.npy").astype(int)
    for seed in F.SEEDS:
        z = np.load(f"{F.CACHE}/{name}__{seed}.npz"); st, mk = z["stats"], z["mask"]; gr = F.R.marginal_ranks(st)
        for j in range(9):
            a = np.flatnonzero((y == 1) & (mk[:, j] > 0))
            if len(a) < 2: continue
            up_full = np.mean((gr[a, j] + 1) / 2); delta = abs(up_full - 0.5)
            for k in F.cells(y):
                sh = F.shots_for(y, seed, k); s = sh[mk[sh, j] > 0]
                if len(s) == 0: continue
                up = np.mean((gr[s, j] + 1) / 2)
                if up == 0.5: continue
                rows.append(dict(k=k, delta=delta, err=int((up > 0.5) != (up_full > 0.5)), bound=min(1.0, np.exp(-2 * k * delta ** 2))))
b = pd.DataFrame(rows); b["delta_bin"] = pd.cut(b.delta, [0, 0.05, 0.1, 0.2, 0.3, 0.5])
bt = b.groupby(["k", "delta_bin"], observed=True).agg(n=("err", "size"), observed_error=("err", "mean"), mean_bound=("bound", "mean")).round(3)
out["bound_violations"] = int((bt.observed_error > bt.mean_bound + 0.02).sum())

with open(f"{F.OUT}/summary.json", "w") as f: json.dump(out, f, indent=1, default=str)
A.round(3).to_csv(f"{F.OUT}/auroc_table.csv"); P.round(3).to_csv(f"{F.OUT}/ap_table.csv"); bt.to_csv(f"{F.OUT}/bound_check.csv")
print(json.dumps(out, indent=1, default=str)); print(A.round(3).to_string()); print(bt.to_string())
