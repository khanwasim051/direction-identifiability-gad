"""Builds the summary tables used in the paper from the per-run result files:
results/tables/all_runs.csv, per_cell_mean_std.csv, wilcoxon.csv, bound_check.csv and
results/heldout/confirmatory_summary.json (claims C1-C3 of CONFIRMATORY_PROTOCOL.md)."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import FDC, RES_R1, RES_DEV, RES_R23, RES_HO, RES_TAB  # noqa: E402
import sys, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(FDC))

RENAME = {"V0_M2": "V0", "GDN_official_last": "GDN", "GDN_official_valsel": "GDN+val",
          "BWGNN_official_last": "BWGNN", "BWGNN_official_valsel": "BWGNN+val"}


def read(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(pattern)) if "check" not in p])


def main():
    s3 = pd.read_csv(f"{RES_R23}/step3.csv")
    dev_f = s3[s3.method.isin(["W1_kgated", "V0_M2"])].replace({"W1_kgated": "FINAL"})
    dev = pd.concat([dev_f[["dataset", "seed", "k", "method", "auroc", "ap"]],
                     read(f"{RES_DEV}/gdn_*.csv"), read(f"{RES_DEV}/bwgnn/bwgnn_*.csv")]); dev["split"] = "development"
    t2 = pd.concat([pd.read_csv(f"{RES_HO}/final_T2.csv").replace({"FINAL_W1": "FINAL"}),
                    read(f"{RES_HO}/gdn_*.csv"), read(f"{RES_HO}/bwgnn/bwgnn_*.csv")]); t2["split"] = "held-out"
    d = pd.concat([dev, t2]); d["method"] = d.method.replace(RENAME)
    d.to_csv(f"{RES_TAB}/all_runs.csv", index=False)
    agg = d.groupby(["split", "dataset", "k", "method"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        n=("auroc", "size")).reset_index()
    agg.to_csv(f"{RES_TAB}/per_cell_mean_std.csv", index=False)

    rows = []
    for split in ["development", "held-out", "all"]:
        a = agg if split == "all" else agg[agg.split == split]
        W = a.pivot_table(index=["split", "dataset", "k"], columns="method", values="auroc_mean")
        for base in ["V0", "GDN", "BWGNN", "GDN+val", "BWGNN+val"]:
            x = W.FINAL - W[base]
            rows.append(dict(split=split, baseline=base, cells=len(x), mean_diff=round(x.mean(), 3), wins=int((x > 0).sum()),
                             p_one_sided=wilcoxon(W.FINAL, W[base], alternative="greater").pvalue))
    pd.DataFrame(rows).to_csv(f"{RES_TAB}/wilcoxon.csv", index=False)

    # confirmatory claims
    H = agg[agg.split == "held-out"].pivot_table(index=["dataset", "k"], columns="method", values="auroc_mean")
    best = H[["GDN", "BWGNN"]].max(axis=1); m = H.mean()
    wins = int((H.FINAL > best).sum())
    summ = dict(C1=bool(m.FINAL >= best.mean() - 0.01), C2=bool(wins >= len(H) / 2), wins=wins, cells=len(H),
                C3=bool(m.FINAL >= m.V0), mean_auroc={k: round(float(v), 4) for k, v in m.items()},
                mean_best_baseline=round(float(best.mean()), 4))
    json.dump(summ, open(f"{RES_HO}/confirmatory_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # direction error of the shot rule against the bounds of Theorem 3 (development graphs)
    import fsdc as F
    rows = []
    for name in F.GRAPHS:
        y = np.load(f"{F.CACHE}/labels_{name}.npy").astype(int)
        for seed in F.SEEDS:
            z = np.load(f"{F.CACHE}/{name}__{seed}.npz"); st, mk = z["stats"], z["mask"]; gr = F.R.marginal_ranks(st)
            for j in range(9):
                a = np.flatnonzero((y == 1) & (mk[:, j] > 0))
                if len(a) < 2: continue
                U = (gr[a, j] + 1) / 2; mu = U.mean(); D = abs(mu - .5); var = U.var()
                for k in F.cells(y):
                    sh = F.shots_for(y, seed, k); s = sh[mk[sh, j] > 0]
                    if len(s) == 0: continue
                    mm = np.mean((gr[s, j] + 1) / 2)
                    if mm == .5: continue
                    rows.append(dict(k=k, D=D, err=int((mm > .5) != (mu > .5)), hoef=min(1, np.exp(-2 * k * D ** 2)),
                                     bern=min(1, np.exp(-k * D ** 2 / (2 * var + 2 * D / 3))) if D > 0 else 1))
    b = pd.DataFrame(rows); b["bin"] = pd.cut(b.D, [0, .05, .1, .2, .3, .5])
    t = b.groupby(["k", "bin"], observed=True).agg(n=("err", "size"), observed=("err", "mean"),
                                                   hoeffding=("hoef", "mean"), bernstein=("bern", "mean")).round(3)
    t.to_csv(f"{RES_TAB}/bound_check.csv")
    print(t.to_string())


if __name__ == "__main__":
    main()
