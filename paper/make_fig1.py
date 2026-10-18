"""Fig. 1: overview of the base detector, drawn as a vector figure at the journal text width (372 pt = 5.15 in),
so that every label is printed at 7 pt or larger. Notation as in Section 3."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import REPO, RES as _RES  # noqa: E402
import pandas as pd
_FIG = REPO / "paper" / "latex" / "fig"; _GEN = REPO / "paper" / "latex" / "gen"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Circle, Rectangle
import numpy as np

INK, INK2, FILL, HEAD, EDGE = "#111111", "#3d3d3d", "#ffffff", "#ebe9e3", "#8a877f"
BLUE, GREY = "#2a78d6", "#9b9890"
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5})
W, H = 5.15, 3.55
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W * 10); ax.set_ylim(0, H * 10); ax.axis("off")
B, S, T = 7.5, 7.0, 8.0          # body, small, box-title font sizes (pt)


def box(x, y, w, h, title=None, th=2.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.5", fc=FILL, ec=EDGE, lw=0.8))
    if title:
        ax.add_patch(FancyBboxPatch((x, y + h - th), w, th, boxstyle="round,pad=0,rounding_size=0.5", fc=HEAD, ec=EDGE, lw=0.8))
        ax.text(x + w / 2, y + h - th / 2, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))


TOP = 32.4
for x, t in ((11.0, "(a) Statistics"), (28.6, "(b) Calibration"), (44.0, "(c) Orientation and fusion")):
    ax.text(x, 34.4, t, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)

# ---- (a) preprocessing
box(0.4, 3.0, 8.6, TOP - 3.0, "Preprocess")
pts = np.array([[2.0, 26.8], [6.6, 27.4], [4.4, 25.0], [1.8, 23.2], [7.2, 23.6], [4.6, 21.2]])
for a, b in [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (2, 5), (1, 4)]:
    ax.plot(*pts[[a, b]].T, color=GREY, lw=0.7, zorder=1)
ax.scatter(*pts.T, s=8, color=INK, zorder=2)
ax.text(4.7, 18.6, r"$(\mathbf{A},\mathbf{X})$", ha="center", fontsize=B)
for i, (s1, s2) in enumerate([("impute by", "median"), ("drop constant", "columns"), ("standardise", "columns"), ("unit-norm", r"rows $\tilde{\mathbf{x}}_i$")]):
    ax.text(4.7, 15.6 - 3.2 * i, s1, ha="center", fontsize=S, color=INK2); ax.text(4.7, 14.2 - 3.2 * i, s2, ha="center", fontsize=S, color=INK2)
arrow(9.0, 17.5, 10.4, 17.5)

# ---- (a) statistics
box(10.4, 13.8, 11.2, TOP - 13.8, "Contextual")
for i, s in enumerate(["node–neighbour disc.", "pair disc. mean", "pair disc. variance", "local closure", "neighbour log-degree", "propagation residual"]):
    y = 27.9 - 2.2 * i
    ax.text(11.0, y, f"$t_{{i{i + 1}}}$", fontsize=B, va="center"); ax.text(12.9, y, s, fontsize=S, va="center", color=INK2)
box(10.4, 3.0, 11.2, 9.8, "Structural")
for i, s in enumerate(["triangle load", "unexpected-edge tail", "motif incoherence"]):
    y = 8.3 - 2.2 * i
    ax.text(11.0, y, f"$t_{{i{i + 7}}}$", fontsize=B, va="center"); ax.text(12.9, y, s, fontsize=S, va="center", color=INK2)
ax.text(17.5, 1.3, r"sampling $q=32$, $L=128$;  undefined values masked ($M_{ij}=0$)", ha="center", fontsize=S, color=INK2)
arrow(21.6, 17.5, 23.0, 17.5); ax.text(22.3, 18.6, "$t_{ij}$", ha="center", fontsize=B)

# ---- (b) neighbourhood exclusion
box(23.0, 3.0, 11.6, TOP - 3.0, "Exclude $i$, $\\mathcal{N}_i$")
cx, cy = 26.9, 20.0
ax.add_patch(Ellipse((cx, cy), 5.4, 8.4, fc="#e9f1fb", ec=BLUE, lw=0.8, ls=(0, (3, 2))))
q = (cx - 0.2, cy); nb = [(cx - 1.2, cy + 2.8), (cx - 1.2, cy - 2.8), (cx + 1.6, cy + 0.2)]
ref = [(32.6, 26.0), (33.2, 20.0), (32.2, 14.6), (24.6, 13.2), (30.0, 12.0)]
for p in nb: ax.plot([q[0], p[0]], [q[1], p[1]], color=GREY, lw=0.7, zorder=1)
for a, b in [(2, 0), (2, 1), (2, 2), (1, 3)]: ax.plot([nb[a][0], ref[b][0]], [nb[a][1], ref[b][1]], color=GREY, lw=0.7, zorder=1)
ax.plot([ref[2][0], ref[4][0]], [ref[2][1], ref[4][1]], color=GREY, lw=0.7, zorder=1)
d = 0.85
ax.add_patch(Polygon([(q[0], q[1] + d), (q[0] + d * .8, q[1]), (q[0], q[1] - d), (q[0] - d * .8, q[1])], fc=INK, zorder=3))
for p in nb: ax.add_patch(Circle(p, 0.55, fc="white", ec=INK, lw=0.8, zorder=3))
for p in ref: ax.add_patch(Rectangle((p[0] - 0.5, p[1] - 0.5), 1.0, 1.0, fc=GREY, ec=INK, lw=0.6, zorder=3))
yl = 9.2
ax.add_patch(Polygon([(24.2, yl + .7), (24.8, yl), (24.2, yl - .7), (23.6, yl)], fc=INK)); ax.text(25.2, yl, "$i$", va="center", fontsize=B)
ax.add_patch(Circle((27.3, yl), 0.5, fc="white", ec=INK, lw=0.8)); ax.text(28.0, yl, "$\\mathcal{N}_i$", va="center", fontsize=B)
ax.add_patch(Rectangle((30.4, yl - .45), 0.9, 0.9, fc=GREY, ec=INK, lw=0.6)); ax.text(31.6, yl, "$\\mathcal{R}_{ij}$", va="center", fontsize=B)
ax.text(28.8, 6.4, r"signed midrank of $t_{ij}$", ha="center", fontsize=S, color=INK2)
ax.text(28.8, 4.6, r"in $\mathcal{R}_{ij}$:  $r_{ij}\in[-1,1]$", ha="center", fontsize=S, color=INK2)
arrow(34.6, 17.5, 36.0, 17.5); ax.text(35.3, 18.6, "$r_{ij}$", ha="center", fontsize=B)

# ---- (c) orientation and fusion
box(36.0, 22.6, 15.0, TOP - 22.6, "Direction $\\theta_j$")
for i, (a, b) in enumerate([("upper", r"$(1+r_{ij})/2$"), ("lower", r"$(1-r_{ij})/2$"), ("two-sided", r"$|r_{ij}|$")]):
    y = 28.4 - 2.2 * i
    ax.text(36.7, y, a, fontsize=S, va="center", color=INK2); ax.text(50.3, y, b, fontsize=B, va="center", ha="right")
arrow(43.5, 22.6, 43.5, 21.0); ax.text(44.0, 21.8, "$e_{ij}$", fontsize=B, va="center")
box(36.0, 13.6, 15.0, 7.4, "Reliability weight")
ax.text(43.5, 16.9, r"$\omega_{ij}=a_{ij}\,b_j\,c_{ij}$", ha="center", fontsize=B + .5)
ax.text(43.5, 14.7, "support, dispersion, stability", ha="center", fontsize=S, color=INK2)
arrow(39.7, 13.6, 39.7, 12.2); arrow(47.3, 13.6, 47.3, 12.2)
box(36.0, 8.6, 7.3, 3.6); ax.text(39.65, 10.4, "contextual $C_i$", ha="center", va="center", fontsize=S)
box(43.7, 8.6, 7.3, 3.6); ax.text(47.35, 10.4, "structural $S_i$", ha="center", va="center", fontsize=S)
arrow(39.65, 8.6, 42.6, 6.9); arrow(47.35, 8.6, 44.4, 6.9)
box(38.0, 3.0, 11.0, 3.9); ax.text(43.5, 4.95, r"$s_i=\max(C_i,\,S_i)$", ha="center", va="center", fontsize=8.5)
fig.savefig(str(_FIG / "Fig1_detector.pdf"))
print("ok")
