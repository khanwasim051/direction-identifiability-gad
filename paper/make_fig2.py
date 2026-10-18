"""Fig. 2 (direction diagnostic heatmap) at the journal text width with larger labels, plus the same values as a
LaTeX table for the appendix (gen/tab_diag.tex)."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import REPO, RES as _RES  # noqa: E402
import pandas as pd
_FIG = REPO / "paper" / "latex" / "fig"; _GEN = REPO / "paper" / "latex" / "gen"
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
_D = pd.read_csv(_RES / "tables" / "direction_diagnostic.csv", index_col=0)
N = {"diag": {g: [None if pd.isna(v) else float(v) for v in r] for g, r in _D.iterrows()}}
INK, INK2, BASE = "#0b0b0b", "#3d3d3d", "#c3c2b7"
BLUE, RED, GRAYMID = "#2a78d6", "#e34948", "#f0efec"
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "stixsans", "font.size": 8})
INJ = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook"]
ZS = ["Reddit", "Amazon", "Disney", "Weibo", "Weibo-ARC", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi"]
T2 = ["CS", "Photo", "Books", "Enron"]; NAME = {"CS": "Coauthor-CS", "Photo": "Amazon-Photo"}
STATS = ["$t_1$ (U)", "$t_2$ (U)", "$t_3$ (U)", "$t_4$ (L)", "$t_5$ (2)", "$t_6$ (U)", "$t_7$ (L)", "$t_8$ (U)", "$t_9$ (U)"]
rows = INJ + ZS + T2
M = np.array([[np.nan if v is None else v for v in (N["diag"][g] or [None] * 9)] for g in rows], float)
cmap = LinearSegmentedColormap.from_list("div", [RED, GRAYMID, BLUE]); cmap.set_bad("#ffffff")
fig, ax = plt.subplots(figsize=(5.15, 6.4))
im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7.8,
                color="#ffffff" if (not np.isnan(v) and abs(v - 0.5) > 0.3) else INK,
                fontweight="bold" if (not np.isnan(v) and v < 0.45) else "normal")
ax.xaxis.tick_top(); ax.set_xticks(range(9)); ax.set_xticklabels(STATS, fontsize=8.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME.get(r, r) for r in rows], fontsize=8.5)
for y0 in (len(INJ) - 0.5, len(INJ) + len(ZS) - 0.5): ax.axhline(y0, color=INK, lw=0.9)
for y, t in (((len(INJ) - 1) / 2, "injected"), (len(INJ) + (len(ZS) - 1) / 2, "organic"), (len(INJ) + len(ZS) + (len(T2) - 1) / 2, "held-out")):
    ax.text(8.62, y, t, rotation=-90, ha="left", va="center", fontsize=8.5, color=INK2)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.03, aspect=45)
cb.set_label("AUROC of the statistic in its fixed direction (0.5: no signal; below 0.5: reversed)", fontsize=8); cb.ax.tick_params(labelsize=7.5)
cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(str(_FIG / "Fig2_direction_diagnostic.pdf")); plt.close(fig)
lines = []
for g, r in zip(rows, M):
    lines.append(NAME.get(g, g) + " & " + " & ".join("--" if np.isnan(v) else (f"\\textbf{{{v:.3f}}}" if v < 0.45 else f"{v:.3f}") for v in r) + " \\\\")
    if g in ("Facebook", "YelpChi"): lines.append("\\midrule")
open(_GEN / "tab_diag.tex", "w").write("\n".join(lines))
print("ok", N["diag"].get("Weibo") is not None)
