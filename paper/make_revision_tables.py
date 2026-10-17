"""Aggregates the revision analyses into CSV summaries and LaTeX tables (paper gen/ folder) and numbers.json."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import RES, RES_TAB, SEALED, REPO  # noqa: E402
import glob, json, numpy as np, pandas as pd
O = str(RES / "revision"); G = str(REPO / "paper" / "latex" / "gen"); NUM = {}
HELD = ["CS", "Photo", "Books", "Enron"]; NAME = {"Weibo-ARC": "Weibo", "CS": "Coauthor-CS", "Photo": "Amazon-Photo"}
f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"

ev = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{O}/eval_*.csv"))])
gd = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{O}/gdn20_*.csv"))])
gd["method"] = gd.method.replace({"GDN_official_last": "GDN", "GDN_official_valsel": "GDN+val"})
d = pd.concat([ev, gd]); d["split"] = np.where(d.dataset.isin(HELD), "held-out", "development")
d.to_csv(f"{O}/rev_all_runs.csv", index=False)
print("graphs:", ev.dataset.nunique(), "gdn graphs:", gd.dataset.nunique(), "seeds:", sorted(d.seed.unique())[:3], len(d.seed.unique()))

def graph_mean(df):  # mean over seeds per (graph,k), then over k per graph, then over graphs
    c = df.groupby(["split", "dataset", "k", "method"])[["auroc", "ap", "nap", "rprec"]].mean().reset_index()
    g = c.groupby(["split", "dataset", "method"])[["auroc", "ap", "nap", "rprec"]].mean().reset_index()
    return c, g

cell, gm = graph_mean(d)
cell.to_csv(f"{O}/rev_cells.csv", index=False)

# ---- Table: 20-seed comparison with lightweight baselines (graph-level means) ----
M = ["FDC", "FDC-R", "GDN", "GDN+val", "BestSingleStat", "PU-logistic", "ShotSim-max", "ShotSim-centroid"]
LAB = {"GDN+val": "GDN$^{\\dagger}$"}
rows = []
for m in M:
    vals = []
    for sp in ("development", "held-out"):
        x = gm[(gm.split == sp) & (gm.method == m)]
        vals += [x.auroc.mean(), x.nap.mean(), x.rprec.mean()]
        NUM.setdefault("light", {})[f"{m}|{sp}"] = [round(float(v), 3) for v in (x.auroc.mean(), x.ap.mean(), x.nap.mean(), x.rprec.mean())]
    rows.append(LAB.get(m, m) + " & " + " & ".join(f3(v) for v in vals) + " \\\\")
open(f"{G}/tab_light.tex", "w").write("\n".join(rows))

# ---- Table: held-out per graph, nAP and R-precision (FDC, FDC-R, BestSingleStat, GDN: 20 seeds; BWGNN 5 seeds nAP) ----
bw = pd.read_csv(RES_TAB / "per_cell_mean_std.csv")
bw = bw[(bw.split == "held-out") & (bw.method == "BWGNN")]
labs = {n: np.load(f"{SEALED}/labels_{n}.npy") for n in HELD}
def nap_from(ap, n, k):
    y = labs[n]; pi = (y.sum() - k) / (len(y) - k); return (ap - pi) / (1 - pi)
bw = bw.assign(nap=[nap_from(r.ap_mean, r.dataset, r.k) for r in bw.itertuples()])
bwg = bw.groupby("dataset")[["auroc_mean", "ap_mean", "nap"]].mean()
rows = []; HM = ["FDC", "FDC-R", "BestSingleStat", "GDN"]
allv = {m: [] for m in HM}; allbw = []
for n in HELD:
    r = gm[(gm.split == "held-out") & (gm.dataset == n)].set_index("method")
    cells = []
    for m in HM:
        v = [r.loc[m, c] if m in r.index else np.nan for c in ("nap", "rprec")]
        cells += v; allv[m].append(v)
    cells.append(bwg.loc[n, "nap"]); allbw.append(bwg.loc[n, "nap"])
    rows.append(NAME.get(n, n) + " & " + " & ".join(f3(v) for v in cells) + " \\\\")
    NUM.setdefault("heldout_metrics", {})[n] = [round(float(v), 3) for v in cells]
mean = sum(([np.mean([x[0] for x in allv[m]]), np.mean([x[1] for x in allv[m]])] for m in HM), []) + [np.mean(allbw)]
rows.append("\\midrule")
rows.append("Mean & " + " & ".join(f3(v) for v in mean) + " \\\\")
NUM["heldout_metrics"]["mean"] = [round(float(v), 3) for v in mean]
open(f"{G}/tab_heldout_metrics.tex", "w").write("\n".join(rows))

# ---- Null experiment ----
nl = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{O}/null_*.csv"))])
nk = nl.groupby("k")[["n_stats", "kept_null", "kept_real", "auroc_null"]].mean()
nk["frac_null"] = nk.kept_null / nk.n_stats
rows = [f"{k} & {r.n_stats:.1f} & {r.kept_null:.1f} ({100 * r.frac_null:.0f}\\%) & {r.kept_real:.1f} & {r.auroc_null:.3f} \\\\" for k, r in nk.iterrows()]
open(f"{G}/tab_null.tex", "w").write("\n".join(rows))
NUM["null"] = {int(k): {c: round(float(v), 3) for c, v in r.items()} for k, r in nk.iterrows()}
NUM["null_auroc_range"] = [round(float(nl.groupby("dataset").auroc_null.mean().min()), 3), round(float(nl.groupby("dataset").auroc_null.mean().max()), 3)]

# ---- Sensitivity: threshold multiplier, gate, p ----
SENS = [("FDC_c0.0", "$c=0$ (no threshold)"), ("FDC_c0.5", "$c=0.5$"), ("FDC", "$c=1$ (FDC)"), ("FDC_c1.5", "$c=1.5$"), ("FDC_c2.0", "$c=2$"),
        ("FDC_gate1", "attributes from $k=1$"), ("FDC", "attributes from $k=3$ (FDC)"), ("FDC_gate4", "attributes from $k=5$"), ("FDC-R", "no attributes (FDC-R)"),
        ("FDC_p8", "$p=8$"), ("FDC", "$p=16$ (FDC)"), ("FDC_p32", "$p=32$")]
rows = []
for i, (m, lab) in enumerate(SENS):
    if i in (5, 9): rows.append("\\midrule")
    vals = []
    for sp in ("development", "held-out"):
        x = gm[(gm.split == sp) & (gm.method == m)]
        vals += [x.auroc.mean(), x.nap.mean()]
    rows.append(lab + " & " + " & ".join(f3(v) for v in vals) + " \\\\")
    NUM.setdefault("sens", {})[lab] = [round(float(v), 3) for v in vals]
open(f"{G}/tab_sens.tex", "w").write("\n".join(rows))

# ---- q, L sensitivity ----
ql = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{O}/qL_*.csv"))])
qc = ql.groupby(["dataset", "q", "L", "k", "method"])[["auroc", "nap"]].mean().reset_index()
qg = qc.groupby(["dataset", "q", "L", "method"])[["auroc", "nap"]].mean().reset_index()
qs = qg.groupby(["q", "L", "method"])[["auroc", "nap"]].mean().unstack("method")
rows = []
for (q, L), r in qs.iterrows():
    rows.append(f"{q} & {L} & " + " & ".join(f3(r[(met, m)]) for m in ("Fixed", "FDC-R", "FDC") for met in ("auroc",)) + " & " + f3(r[("nap", "FDC")]) + " \\\\")
open(f"{G}/tab_qL.tex", "w").write("\n".join(rows))
NUM["qL"] = {f"{q},{L}": {m: round(float(r[("auroc", m)]), 3) for m in ("Fixed", "FDC-R", "FDC")} for (q, L), r in qs.iterrows()}
NUM["qL_graphs"] = sorted(ql.dataset.unique().tolist())
NUM["qL_maxdev"] = round(float(qg.pivot_table(index=["dataset", "method"], columns=["q", "L"], values="auroc").apply(lambda r: (r - r[(32, 128)]).abs().max(), axis=1).max()), 3)

json.dump(NUM, open(f"{O}/rev_numbers.json", "w"), indent=1)
print(json.dumps(NUM, indent=1)[:4000])
