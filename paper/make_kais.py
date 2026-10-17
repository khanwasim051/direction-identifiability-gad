"""Generates every table (paper/latex/gen/*.tex), the AUROC-vs-k figure and gen/numbers.json for the manuscript from
the result files in results/. No number in the tables is typed by hand. dataset_statistics.tex holds the graph
sizes and anomaly counts (computed by the loaders in gad/bench_data.py)."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import REPO, RES_R1, RES_R23, RES_HO, RES_TAB  # noqa: E402
import os, json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LATEX = REPO / "paper" / "latex"; os.chdir(LATEX); G = "gen"; NUM = {}
cell = pd.read_csv(f"{RES_TAB}/per_cell_mean_std.csv")
cell["method"] = cell.method.replace({"FINAL": "FDC", "V0": "FDC-R"})
INJ = {"Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook", "CS", "Photo"}
NAME = {"Weibo-ARC": "Weibo", "CS": "Coauthor-CS", "Photo": "Amazon-Photo"}
DEV_ORDER = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook",
             "Reddit", "Amazon", "Disney", "Weibo-ARC", "Tolokers", "Amazon-all", "YelpChi", "YelpChi-all", "Questions"]
T2_ORDER = ["CS", "Photo", "Books", "Enron"]
METH = ["FDC", "FDC-R", "GDN", "BWGNN", "GDN+val", "BWGNN+val"]
f3 = lambda x: f"{x:.3f}"

# fixed-direction reference (M0) on development graphs, same shots excluded
ours = pd.read_csv(f"{RES_R1}/ours.csv")
m0 = ours[ours.method == "M0_frozen"].groupby(["dataset", "k"]).agg(auroc_mean=("auroc", "mean"), ap_mean=("ap", "mean")).reset_index()
m0["method"] = "Fixed"; m0["split"] = "development"
cellx = pd.concat([cell, m0], ignore_index=True)

def piv(split, val="auroc_mean"):
    return cellx[cellx.split == split].pivot_table(index=["k", "dataset"], columns="method", values=val)

# ---------------- Table: main results by k (development) ----------------
A, P = piv("development"), piv("development", "ap_mean")
A["best_a"] = A[["GDN", "BWGNN"]].max(axis=1)
rows = []
for k in (1, 3, 5, 10):
    a, p = A.loc[k], P.loc[k]
    wins = int((a.FDC > a.best_a).sum())
    NUM[f"dev_k{k}"] = {m: round(float(a[m].mean()), 3) for m in ["Fixed"] + METH} | {"wins": wins, "cells": len(a),
                         "ap": {m: round(float(p[m].mean()), 3) for m in ["Fixed"] + METH}}
    vals = [a[m].mean() for m in ["Fixed", "FDC-R", "FDC", "GDN", "BWGNN", "GDN+val", "BWGNN+val"]]
    best = max(vals[:5])
    s = [("\\textbf{%s}" % f3(v)) if (i < 5 and abs(v - best) < 1e-12) else f3(v) for i, v in enumerate(vals)]
    rows.append(f"{k} & " + " & ".join(s) + f" & {wins}/{len(a)} \\\\")
rows.append("\\midrule")
for k in (1, 3, 5, 10):
    p = P.loc[k]; vals = [p[m].mean() for m in ["Fixed", "FDC-R", "FDC", "GDN", "BWGNN", "GDN+val", "BWGNN+val"]]
    best = max(vals[:5])
    s = [("\\textbf{%s}" % f3(v)) if (i < 5 and abs(v - best) < 1e-12) else f3(v) for i, v in enumerate(vals)]
    rows.append(f"{k} & " + " & ".join(s) + " & -- \\\\")
open(f"{G}/tab_main.tex", "w").write("\n".join(rows))

# ---------------- Table: per-graph development, k = 10 (Disney: k = 3) ----------------
C = cell[cell.split == "development"].set_index(["dataset", "k", "method"])
rows = []
for d in DEV_ORDER:
    k = 3 if d == "Disney" else 10
    r = C.loc[(d, k)]
    fx = m0[(m0.dataset == d) & (m0.k == k)].auroc_mean.iloc[0]
    def ms(m):
        return f"{r.loc[m, 'auroc_mean']:.3f}\\,{{\\tiny$\\pm${r.loc[m, 'auroc_std']:.3f}}}"
    best = max(r.loc[m, "auroc_mean"] for m in ["FDC", "GDN", "BWGNN"])
    cells = []
    for m in ["FDC", "GDN", "BWGNN"]:
        txt = ms(m)
        if abs(r.loc[m, "auroc_mean"] - best) < 1e-12: txt = "\\textbf{" + f"{r.loc[m, 'auroc_mean']:.3f}" + "}" + txt[5:]
        cells.append(txt)
    lab = NAME.get(d, d) + ("$^{\\ast}$" if d == "Disney" else "")
    grp = "inj." if d in INJ else "org."
    rows.append(f"{lab} & {grp} & {f3(fx)} & {f3(r.loc['FDC-R','auroc_mean'])} & " + " & ".join(cells) +
                f" & {f3(r.loc['GDN+val','auroc_mean'])} & {f3(r.loc['BWGNN+val','auroc_mean'])} \\\\")
open(f"{G}/tab_pergraph.tex", "w").write("\n".join(rows))

# ---------------- Table: held-out confirmatory ----------------
H = cell[cell.split == "held-out"].pivot_table(index=["dataset", "k"], columns="method", values="auroc_mean")
HA = cell[cell.split == "held-out"].pivot_table(index=["dataset", "k"], columns="method", values="ap_mean")
H["best_a"] = H[["GDN", "BWGNN"]].max(axis=1)
rows = []
for d in T2_ORDER:
    for k in sorted(H.loc[d].index):
        r = H.loc[(d, k)]
        best = max(r[m] for m in ["FDC", "FDC-R", "GDN", "BWGNN"])
        s = [("\\textbf{%s}" % f3(r[m])) if abs(r[m] - best) < 1e-12 else f3(r[m]) for m in ["FDC", "FDC-R", "GDN", "BWGNN"]]
        rows.append(f"{NAME.get(d, d)} & {k} & " + " & ".join(s) + f" & {f3(r['GDN+val'])} & {f3(r['BWGNN+val'])} \\\\")
rows.append("\\midrule")
rows.append("Mean AUROC & & " + " & ".join(f3(H[m].mean()) for m in ["FDC", "FDC-R", "GDN", "BWGNN", "GDN+val", "BWGNN+val"]) + " \\\\")
rows.append("Mean AP & & " + " & ".join(f3(HA[m].mean()) for m in ["FDC", "FDC-R", "GDN", "BWGNN", "GDN+val", "BWGNN+val"]) + " \\\\")
open(f"{G}/tab_heldout.tex", "w").write("\n".join(rows))
NUM["heldout"] = {m: round(float(H[m].mean()), 3) for m in METH + ["best_a"]} | {
    "ap": {m: round(float(HA[m].mean()), 3) for m in METH}, "cells": len(H),
    "wins_vs_best_a": int((H.FDC > H.best_a).sum()),
    "wins_vs_best_val": int((H.FDC > H[["GDN+val", "BWGNN+val"]].max(axis=1)).sum())}

# ---------------- Table: Wilcoxon ----------------
W = pd.read_csv(f"{RES_TAB}/wilcoxon.csv")
W["baseline"] = W.baseline.replace({"V0": "FDC-R"})
order = ["GDN", "BWGNN", "GDN+val", "BWGNN+val", "FDC-R"]
lab = {"GDN": "GDN", "BWGNN": "BWGNN", "GDN+val": "GDN$^{\\dagger}$", "BWGNN+val": "BWGNN$^{\\dagger}$", "FDC-R": "FDC-R"}
def pfmt(p):
    if p < 1e-4:
        e = int(np.floor(np.log10(p))); return f"$<10^{{{e + 1}}}$"
    return f"{p:.4f}"
rows = []
for b in order:
    parts = []
    for sp in ("held-out", "development", "all"):
        r = W[(W.split == sp) & (W.baseline == b)].iloc[0]
        parts += [f"{r.mean_diff:+.3f}", f"{int(r.wins)}/{int(r.cells)}", pfmt(r.p_one_sided)]
    rows.append(lab[b] + " & " + " & ".join(parts) + " \\\\")
open(f"{G}/tab_wilcoxon.tex", "w").write("\n".join(rows))
NUM["wilcoxon"] = W.to_dict(orient="records")

# ---------------- Table: ablation / development history ----------------
s2 = pd.read_csv(f"{RES_R23}/step2.csv"); s3 = pd.read_csv(f"{RES_R23}/step3.csv")
abl = pd.concat([ours[ours.method.isin(["M0_frozen", "M1_DIR"])][["dataset", "seed", "k", "method", "auroc"]],
                 s2[s2.method.isin(["V0_M2", "V1_pooled", "V2_shrink", "V3_top8"])][["dataset", "seed", "k", "method", "auroc"]],
                 s3[s3.method.isin(["W1_kgated", "W2_maxfusion"])][["dataset", "seed", "k", "method", "auroc"]]])
cm = abl.groupby(["method", "dataset", "k"]).auroc.mean().reset_index()
cm["grp"] = np.where(cm.dataset.isin(INJ), "inj", "org")
desc = [("M0_frozen", "Fixed directions (no shots)", "--"),
        ("M1_DIR", "Shot-calibrated directions, relational only", "R1"),
        ("V0_M2", "+ separation weights (FDC-R)", "R1"),
        ("V1_pooled", "+ attribute statistics, pooled", "R2"),
        ("V2_shrink", "+ null-shrinkage $\\tau_k$", "R2"),
        ("V3_top8", "+ top-8 selection", "R2"),
        ("W2_maxfusion", "Max of FDC-R and attribute score", "R3"),
        ("W1_kgated", "$k$-gated: FDC-R if $k<3$, else pooled+shrinkage (FDC)", "R3")]
rows = []
for m, d, rd in desc:
    x = cm[cm.method == m]
    byk = x.groupby("k").auroc.mean()
    vals = [x.auroc.mean(), x[x.grp == "inj"].auroc.mean(), x[x.grp == "org"].auroc.mean()] + [byk[k] for k in (1, 3, 5, 10)]
    rows.append(f"{d} & {rd} & " + " & ".join(f3(v) for v in vals) + " \\\\")
    NUM.setdefault("ablation", {})[m] = [round(float(v), 4) for v in vals]
open(f"{G}/tab_ablation.tex", "w").write("\n".join(rows))

# ---------------- Table: runtime ----------------
rt = pd.read_csv(f"{RES_TAB}/runtime.csv").sort_values("edges")
rows = []
for _, r in rt.iterrows():
    rows.append(f"{NAME.get(r.dataset, r.dataset)} & {r.nodes:,} & {r.edges:,} & {r.final_statistics_s + r.final_attribute_s:.2f} & "
                f"{r.final_calibration_s:.3f} & {r.gdn_official_s:.2f} \\\\")
open(f"{G}/tab_runtime.tex", "w").write("\n".join(rows))
NUM["runtime"] = {"cal_max": float(rt.final_calibration_s.max()), "stat_min": float((rt.final_statistics_s + rt.final_attribute_s).min()),
                  "stat_max": float((rt.final_statistics_s + rt.final_attribute_s).max())}

# ---------------- Table: datasets ----------------
src = open(REPO / "paper" / "dataset_statistics.tex").read().strip().split("\n")
rows = []
for line in src:
    c = [x.strip() for x in line.rstrip("\\ ").split("&")]
    if c[0] == "Weibo (UNP)": continue
    if c[0] == "Weibo (ARC)": c[0] = "Weibo"
    split = "Held-out" if c[0] in ("Coauthor-CS", "Amazon-Photo", "Books", "Enron") else "Dev."
    rows.append(" & ".join(c[:7] + [split]) + " \\\\")
dev = [r for r in rows if r.endswith("Dev. \\\\")]; ho = [r for r in rows if r.endswith("Held-out \\\\")]
open(f"{G}/tab_datasets.tex", "w").write("\n".join(dev + ["\\midrule"] + ho))

# ---------------- Bound-check table is in the theory section (from paper_tables/bound_check.csv) ----------------

# ---------------- Figure: AUROC vs k ----------------
col = {"FDC": "#2a78d6", "GDN": "#eb6834", "BWGNN": "#1baf7a", "FDC-R": "#e87ba4"}
mk = {"FDC": "o", "GDN": "s", "BWGNN": "^", "FDC-R": "D"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, sp, title in ((axes[0], "development", "(a) Development graphs (16)"), (axes[1], "held-out", "(b) Held-out graphs (4)")):
    Q = cell[cell.split == sp].groupby(["method", "k"]).auroc_mean.mean().unstack()
    ks = [1, 3, 5, 10]
    for m in ["FDC", "FDC-R", "GDN", "BWGNN"]:
        ax.plot(ks, Q.loc[m, ks], color=col[m], marker=mk[m], ms=6, lw=2, label=m, clip_on=False)
        if m in ("GDN", "BWGNN"):
            ax.plot(ks, Q.loc[m + "+val", ks], color=col[m], marker=mk[m], ms=5, lw=1.5, ls="--", mfc="white",
                    label=m + " + labelled val.")
    ax.axhline(0.5, color="#9a9a96", lw=1, ls=":")
    ax.set_xticks(ks); ax.set_xlabel("labelled anomalies $k$"); ax.set_title(title, fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("mean AUROC"); axes[0].set_ylim(0.45, 0.85)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("fig/Fig3_auroc_vs_k.pdf", bbox_inches="tight"); fig.savefig("fig/Fig3_auroc_vs_k.png", dpi=200, bbox_inches="tight")

json.dump(NUM, open(f"{G}/numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in NUM.items() if k != "wilcoxon"}, indent=1, default=float))
