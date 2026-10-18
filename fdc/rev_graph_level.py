"""Graph-level significance and hierarchical-bootstrap effect sizes (reviewer comment 3).
Unit of analysis = graph. Per graph, the effect is the mean over its (k, seed) runs of FDC minus the comparison method
(paired on k and seed). Held-out: exact one-sided sign test over graphs. Development: one-sided Wilcoxon over graphs.
CI: hierarchical bootstrap (resample graphs, then seeds within each graph), 10,000 replicates, percentile 95%."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES, RES_TAB  # noqa: E402
import numpy as np, pandas as pd, json
from scipy.stats import wilcoxon, binomtest

d = pd.read_csv(RES_TAB / "all_runs.csv")
d["method"] = d.method.replace({"FINAL": "FDC", "V0": "FDC-R"})
COMP = ["GDN", "BWGNN", "GDN+val", "BWGNN+val", "FDC-R"]
rng = np.random.default_rng(2026)


def diffs(split, base, metric="auroc"):
    x = d[d.split == split].pivot_table(index=["dataset", "k", "seed"], columns="method", values=metric)
    return (x["FDC"] - x[base]).rename("diff").reset_index()


def boot(df, B=10000):
    g = {name: grp.groupby("seed")["diff"].apply(np.array).to_dict() for name, grp in df.groupby("dataset")}
    names = list(g); out = np.empty(B)
    for b in range(B):
        gs = rng.choice(names, len(names), replace=True); vals = []
        for name in gs:
            seeds = list(g[name]); ss = rng.choice(seeds, len(seeds), replace=True)
            vals.append(np.mean(np.concatenate([g[name][s] for s in ss])))
        out[b] = np.mean(vals)
    return np.percentile(out, [2.5, 97.5])


rows, per_graph, paired = [], [], []
for split in ("held-out", "development"):
    for base in COMP:
        for metric in ("auroc", "ap"):
            df = diffs(split, base, metric)
            g = df.groupby("dataset")["diff"].mean()
            wins = int((g > 0).sum())
            if split == "held-out":
                p = binomtest(wins, len(g), alternative="greater").pvalue; test = "sign"
            else:
                p = wilcoxon(g, alternative="greater").pvalue; test = "wilcoxon"
            lo, hi = boot(df)
            paired.append(df.assign(split=split, baseline=base, metric=metric))
            rows.append(dict(split=split, baseline=base, metric=metric, graphs=len(g), wins=wins, mean_diff=g.mean(),
                             ci_low=lo, ci_high=hi, test=test, p=p))
            if True:
                for name, v in g.items(): per_graph.append(dict(split=split, baseline=base, metric=metric, dataset=name, diff=v))
R = pd.DataFrame(rows); R.to_csv(RES / "revision" / "graph_level.csv", index=False)
pd.DataFrame(per_graph).to_csv(RES / "revision" / "graph_level_per_graph.csv", index=False)
pd.concat(paired).to_csv(RES / "revision" / "graph_level_paired_runs.csv", index=False)
print(R.to_string())
