import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
N = json.load(open(RES_PAPER / "numbers.json")); FIG = str(PAPER / "latex" / "fig") + "/"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED, GRAYMID = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#f0efec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
INJ = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook"]
ZS = ["Reddit", "Amazon", "Disney", "Weibo", "Weibo-ARC", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi"]
T2 = ["CS", "Photo", "Books", "Enron"]
STATS = ["$t_1$ node–nbr disc. (U)", "$t_2$ pair mean (U)", "$t_3$ pair var. (U)", "$t_4$ closure (L)", "$t_5$ nbr degree (2)",
         "$t_6$ prop. residual (U)", "$t_7$ triangle load (L)", "$t_8$ unexp. edge (U)", "$t_9$ motif incoh. (U)"]
# ---- Fig 2: direction diagnostic heatmap
rows = INJ + ZS + T2; M = np.array([[np.nan if v is None else v for v in (N["diag"][g] or [None]*9)] for g in rows], float)
cmap = LinearSegmentedColormap.from_list("div", [RED, GRAYMID, BLUE]); cmap.set_bad("#ffffff")
fig, ax = plt.subplots(figsize=(6.85, 6.9))
im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]; ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if (not np.isnan(v) and abs(v - 0.5) > 0.3) else INK, fontweight="bold" if (not np.isnan(v) and v < 0.45) else "normal")
ax.set_xticks(range(9)); ax.set_xticklabels(STATS, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
for y0 in (len(INJ) - 0.5, len(INJ) + len(ZS) - 0.5): ax.axhline(y0, color=INK, lw=0.8)
ax.text(8.65, (len(INJ) - 1) / 2, "injected", rotation=-90, ha="left", va="center", fontsize=7, color=INK2)
ax.text(8.65, len(INJ) + (len(ZS) - 1) / 2, "organic", rotation=-90, ha="left", va="center", fontsize=7, color=INK2)
ax.text(8.65, len(INJ) + len(ZS) + (len(T2) - 1) / 2, "untouched (T2)", rotation=-90, ha="left", va="center", fontsize=7, color=INK2)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.16, aspect=40); cb.set_label("AUROC of oriented coordinate (0.5 = no signal; <0.5 = direction reversed)", fontsize=7)
cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(FIG + "Fig2_direction_diagnostic.pdf"); plt.close(fig)
# ---- Fig 3: rule study
R = N["rules"]; cands = list(R["dev"])
skew = [c for c in cands if c.startswith(("A_", "B_"))]
best_skew = max(skew, key=lambda c: R["dev"][c]["score"])
fig, axs = plt.subplots(1, 2, figsize=(6.85, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axs[0]; splits = [("dev", "Development"), ("t1", "T1 (seen)"), ("t2", "T2 (untouched)")]
LBL = {"dev": "Development", "t1": "T1\n(seen)", "t2": "T2\n(untouched)"}
series = [("F0_frozen", "Fixed directions", BLUE, "o"), ("C_consensus_frozen", "Consensus rule (selected)", ORANGE, "s"),
          ("F1_twosided", "All two-sided", AQUA, "D"), (best_skew, "Best skewness rule", INK2, "^")]
for i, (c, lab, col, mk) in enumerate(series):
    xs = np.arange(3) + (i - 1.5) * 0.12
    ax.plot(xs, [R[s][c]["auroc"] for s, _ in splits], marker=mk, ms=6, lw=0, color=col, label=lab, mec="#fcfcfb", mew=1)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3))); ax.text(2.45, 0.505, "random", fontsize=6.5, color=MUTED, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([LBL[s] for s, _ in splits]); ax.set_ylabel("Mean AUROC over units"); ax.set_ylim(0.36, 0.76)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); ax.set_title("(a) Pre-registered rules by split", fontsize=8, loc="left", color=INK)
ax = axs[1]; items = []
for s, lab in splits:
    for u, v in R[s]["C_consensus_frozen"]["per_unit"].items():
        items.append((u.replace("syn:", "syn. "), v["auroc"] - R[s]["F0_frozen"]["per_unit"][u]["auroc"], s))
items.sort(key=lambda t: t[1]); cols = {"dev": BLUE, "t1": ORANGE, "t2": AQUA}
y = np.arange(len(items))
ax.barh(y, [d for _, d, _ in items], color=[cols[s] for _, _, s in items], height=0.7, edgecolor="#fcfcfb", linewidth=0.5)
ax.set_yticks(y); ax.set_yticklabels([u for u, _, _ in items], fontsize=6); ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("AUROC change, consensus rule − fixed directions"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cols[s], label=l) for s, l in splits], frameon=False, fontsize=6.5, loc="upper left")
ax.set_title("(b) Per-unit effect of the selected rule", fontsize=8, loc="left", color=INK)
fig.tight_layout(); fig.savefig(FIG + "Fig3_rule_study.pdf"); plt.close(fig)
# ---- Fig 4: direction recovery
rec = N["recovery"]
order = ["Skewness rule A(kelly,0.1)", "Fixed (frozen) directions", "Consensus rule C", "1 normal shots", "5 normal shots", "10 normal shots",
         "1 anomaly + 1 normal shots", "3 anomaly + 3 normal shots", "5 anomaly + 5 normal shots", "10 anomaly + 10 normal shots"]
lab = {"Skewness rule A(kelly,0.1)": "Skewness rule", "Fixed (frozen) directions": "Fixed directions", "Consensus rule C": "Consensus rule",
       "1 normal shots": "1 normal", "5 normal shots": "5 normal", "10 normal shots": "10 normal",
       "1 anomaly + 1 normal shots": "1 anomaly + 1 normal", "3 anomaly + 3 normal shots": "3 anomaly + 3 normal",
       "5 anomaly + 5 normal shots": "5 anomaly + 5 normal", "10 anomaly + 10 normal shots": "10 anomaly + 10 normal"}
grp = lambda o: ("Unlabeled only", BLUE, "o") if "shots" not in o else (("Normal shots only", ORANGE, "s") if "anomaly" not in o else ("Anomaly + normal shots", AQUA, "D"))
fig, ax = plt.subplots(figsize=(6.85, 2.8))
seen = set()
for i, o in enumerate(order):
    g, col, mk = grp(o); v = rec[o]["pooled"]
    ax.plot([0.5, v], [i, i], color=GRID, lw=2, solid_capstyle="round")
    ax.plot(v, i, marker=mk, ms=7, color=col, mec="#fcfcfb", mew=1, lw=0, label=None if g in seen else g); seen.add(g)
    ax.text(v + 0.012 if v >= 0.5 else v - 0.012, i, f"{v:.2f}", va="center", ha="left" if v >= 0.5 else "right", fontsize=6.5, color=INK)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3))); ax.text(0.505, -0.75, "chance", fontsize=6.5, color=MUTED, ha="left")
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[o] for o in order], fontsize=7)
ax.set_xlim(0.2, 0.85); ax.set_xlabel("Share of decisive statistics oriented in the anomalous direction (pooled over graphs and seeds)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(len(order) - 0.5, -1.1); ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(FIG + "Fig4_direction_recovery.pdf"); plt.close(fig)
print("best skew rule on dev:", best_skew)
