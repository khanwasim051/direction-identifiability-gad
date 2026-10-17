"""Graph-level significance table (row groups: held-out, development)."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import RES, REPO  # noqa: E402
import pandas as pd, numpy as np
R = pd.read_csv(RES / "revision" / "graph_level.csv")
lab = {"GDN": "GDN", "BWGNN": "BWGNN", "GDN+val": "GDN$^{\\dagger}$", "BWGNN+val": "BWGNN$^{\\dagger}$", "FDC-R": "FDC-R"}
def pf(p): return f"{p:.4f}" if p >= 1e-4 else f"$<10^{{{int(np.floor(np.log10(p))) + 1}}}$"
rows = []
for sp, title in (("held-out", "Held-out graphs (4; exact sign test)"), ("development", "Development graphs (16; Wilcoxon signed-rank test)")):
    if rows: rows.append("\\midrule")
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
    for b in ["GDN", "BWGNN", "GDN+val", "BWGNN+val", "FDC-R"]:
        parts = []
        for met in ("auroc", "ap"):
            r = R[(R.split == sp) & (R.baseline == b) & (R.metric == met)].iloc[0]
            parts.append(f"{r.mean_diff:+.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] & {int(r.wins)}/{int(r.graphs)} & {pf(r.p)}")
        rows.append("\\quad " + lab[b] + " & " + " & ".join(parts) + " \\\\")
open(REPO / "paper" / "latex" / "gen" / "tab_graphlevel.tex", "w").write("\n".join(rows))
print("\n".join(rows))
