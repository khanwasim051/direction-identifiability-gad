import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import json, numpy as np
N = json.load(open(RES_PAPER / "numbers.json")); OUT = str(PAPER / "latex" / "gen") + "/"
B = N["baselines"]; core = N["core"]
f3 = lambda v: "--" if v is None else f"{v:.3f}"
def bold_row_max(vals):
    ok = [v for v in vals if v is not None]; mx = max(ok) if ok else None
    return [("\\textbf{" + f3(v) + "}") if (v is not None and mx is not None and abs(v - mx) < 1e-12) else f3(v) for v in vals]
INJ = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook"]
ZS = ["Reddit", "Amazon", "Disney", "Weibo", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi"]
T2 = ["CS", "Photo", "Books", "Enron"]
SHORT = {"BlogCatalog": "BlogC.", "CiteSeer": "CiteS.", "Facebook": "Faceb.", "Amazon-all": "Amz-all", "YelpChi-all": "Yelp-all",
         "Questions": "Quest.", "Tolokers": "Tolok.", "YelpChi": "YelpChi", "PubMed": "PubMed"}
# ---- datasets table
TYPE = {**{g: "injected" for g in ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "CS", "Photo"]}, "Facebook": "injected",
        **{g: "organic" for g in ["Reddit", "Amazon", "Disney", "Weibo", "Weibo-ARC", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi", "Books", "Enron"]}}
SPLIT = {**{g: "Dev" for g in ["Cora", "ACM", "BlogCatalog", "Amazon", "Amazon-all", "Tolokers"]}, **{g: "T1" for g in ["CiteSeer", "PubMed", "Flickr", "Facebook", "Reddit", "Disney", "Weibo", "YelpChi", "YelpChi-all", "Questions"]},
         **{g: "T2" for g in T2}, "Weibo-ARC": "--"}
SRC = {**{g: "ARC" for g in ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook", "Reddit", "Amazon", "Tolokers", "Questions", "YelpChi", "Weibo-ARC", "CS", "Photo"]},
       "Disney": "UNPrompt", "Weibo": "UNPrompt", "Amazon-all": "CARE-GNN", "YelpChi-all": "CARE-GNN", "Books": "PyGOD", "Enron": "PyGOD"}
NAME = {"CS": "Coauthor-CS", "Photo": "Amazon-Photo", "Weibo-ARC": "Weibo (ARC)", "Weibo": "Weibo (UNP)"}
L = []
for g in INJ + ZS + ["Weibo-ARC"] + T2:
    s = N["graphs"][g]
    L.append(f"{NAME.get(g, g)} & {s['nodes']:,} & {s['edges']:,} & {s['features']:,} & {s['anomalies']:,} ({100*s['rate']:.2f}\\%) & {TYPE[g][:3]}. & {SRC[g]} & {SPLIT[g]} \\\\")
open(OUT + "tab_datasets.tex", "w").write("\n".join(L))
# ---- benchmark tables
def bench(ds, rows, k, avg_rows):
    out = []
    full_avg = {r: (np.mean([g(r, d, k) for d in ds]) if all(g(r, d, k) is not None for d in ds) else None) for r, _ in rows}
    executed = [r for r, ex in rows if ex]
    for r, ex in rows:
        vals = [g(r, d, k) for d in ds]
        cells = []
        for d, v in zip(ds, vals):
            best = max((g(e, d, k) for e in executed if g(e, d, k) is not None), default=None)
            cells.append(("\\textbf{" + f3(v) + "}") if (ex and v is not None and best is not None and abs(v - best) < 1e-12) else f3(v))
        out.append(f"{label(r)} & " + " & ".join(cells) + f" & {f3(full_avg[r]) if r in avg_rows else '--'} \\\\")
    return out
def g(r, d, k):
    key = {0: "auroc", 1: "ap"}[k]
    if r.startswith("core:"): return core[d][r[5:]][key] if d in core else None
    v = B.get(r, {}).get(d); return None if v is None else v[0 if k == 0 else 2]
LABEL = {"core:frozen": "Ours, fixed dir.", "core:twosided": "\\quad two-sided$^{*}$", "core:no_triangle": "\\quad no triangles$^{*}$",
         "NeighborDiv": "NeighborDiv", "FreeGAD": "FreeGAD", "UNPrompt (rerun)": "UNPrompt$^{\\dagger}$", "AnomalyGFM (rerun)": "AnomalyGFM$^{\\dagger}$",
         "UNPrompt (published)": "UNPrompt (pub.)", "AnomalyGFM (published)": "AnomalyGFM (pub.)",
         "DOMINANT (published, per-graph)": "DOMINANT (pub.)", "CoLA (published, per-graph)": "CoLA (pub.)", "TAM (published, per-graph)": "TAM (pub.)"}
label = lambda r: LABEL[r]
zs_rows = [("NeighborDiv", 1), ("FreeGAD", 1), ("core:frozen", 1), ("UNPrompt (rerun)", 1), ("AnomalyGFM (rerun)", 1), ("UNPrompt (published)", 0), ("AnomalyGFM (published)", 0)]
inj_rows = [("NeighborDiv", 1), ("FreeGAD", 1), ("core:frozen", 1), ("core:twosided", 0), ("core:no_triangle", 0), ("DOMINANT (published, per-graph)", 0), ("CoLA (published, per-graph)", 0), ("TAM (published, per-graph)", 0)]
for name, ds, rows, avg in (("zs", ZS, zs_rows, ["NeighborDiv", "FreeGAD", "core:frozen", "UNPrompt (rerun)"]), ("inj", INJ, inj_rows, ["NeighborDiv", "FreeGAD", "core:frozen", "core:twosided", "core:no_triangle"])):
    for k, m in ((0, "auroc"), (1, "ap")):
        open(OUT + f"tab_{name}_{m}.tex", "w").write("\n".join(bench(ds, rows, k, avg)))
open(OUT + "hdr_zs.tex", "w").write(" & ".join(SHORT.get(d, d) for d in ZS))
open(OUT + "hdr_inj.tex", "w").write(" & ".join(SHORT.get(d, d) for d in INJ))
# ---- rule table
R = N["rules"]
def cname(c):
    if c == "F0_frozen": return "F0: fixed directions"
    if c == "F1_twosided": return "F1: all two-sided"
    if c == "C_consensus_frozen": return "C: consensus (selected)"
    if c.startswith("D_"): return "D: consensus from A"
    fam, kind, t = c.split("_"); return f"{fam}: {'moment' if kind=='moment' else 'Kelly'}, $\\tau={t}$"
order = sorted(R["dev"], key=lambda c: -R["dev"][c]["score"])
L = []
for c in order:
    L.append(cname(c) + " & " + " & ".join(f"{R[s][c]['score']:.3f} & {R[s][c]['auroc']:.3f} & {R[s][c]['ap']:.3f}" for s in ("dev", "t1", "t2")) + " \\\\")
open(OUT + "tab_rules.tex", "w").write("\n".join(L))
# ---- few-shot tables
F = N["fewshot"]; fs = lambda m, k, d, key="auroc": F.get(f"{m}|{k}|{d}", {}).get(key)
ARC = {"Cora": (.8745,), "CiteSeer": (.9095,), "ACM": (.7988,), "BlogCatalog": (.7476,), "Facebook": (.6756,), "Weibo-ARC": (.8885,), "Reddit": (.6004,), "Amazon": (.8067,)}
L = []
cols = []
for d, (arc,) in ARC.items():
    row = [fs("frozen_zero_shot", 10, d), fs("normal_shot_excess_tail", 10, d), arc, fs("anomaly+normal_shots", 10, d)]
    cols.append(row)
    nm = {"Weibo-ARC": "Weibo (ARC)"}.get(d, d)
    best = max(row[:3])
    L.append(nm + " & " + " & ".join(("\\textbf{" + f3(v) + "}") if (i < 3 and abs(v - best) < 1e-12) else f3(v) for i, v in enumerate(row)) + " \\\\")
m = np.mean(cols, 0); L.append("\\midrule\nMean & " + " & ".join(f3(v) for v in m) + " \\\\")
open(OUT + "tab_fs_arc.tex", "w").write("\n".join(L))
AG = {1: [.5922, .8531, .6649, .8972, .5872, .5898, .5303], 5: [.6023, .8600, .6613, .9011, .5951, .6095, .5426], 10: [.6252, .8649, .6649, .9215, .6064, .6140, .5611]}
DS = ["Reddit", "Amazon", "Disney", "Amazon-all", "YelpChi-all", "Tolokers", "Questions"]
L = []
for k in (1, 5, 10):
    for lab, vals in ((f"Ours, normal shots", [fs("normal_shot_excess_tail", k, d) for d in DS]), ("AnomalyGFM (pub.)", AG[k])):
        L.append((f"\\multirow{{2}}{{*}}{{{k}}}" if lab.startswith("Ours") else "") + f" & {lab} & " + " & ".join(f3(v) for v in vals) + f" & {np.mean(vals):.3f} \\\\")
    if k != 10: L.append("\\cmidrule(l){2-10}")
L.append("\\midrule\n0 & Ours, fixed & " + " & ".join(f3(fs("frozen_zero_shot", 10, d)) for d in DS) + f" & {np.mean([fs('frozen_zero_shot', 10, d) for d in DS]):.3f} \\\\")
an = [fs("anomaly+normal_shots", 10, d) for d in DS]
L.append("10+10$^{\\ddagger}$ & Ours, anom.+norm. & " + " & ".join(f3(v) for v in an) + " & -- \\\\")
open(OUT + "tab_fs_agfm.tex", "w").write("\n".join(L))
# ---- key numbers for text
K = {"zs_core_avg": np.mean([core[d]["frozen"]["auroc"] for d in ZS]), "zs_unp_avg": np.mean([B["UNPrompt (rerun)"][d][0] for d in ZS]),
     "inj_core_avg": np.mean([core[d]["frozen"]["auroc"] for d in INJ]), "inj_fg_avg": np.mean([B["FreeGAD"][d][0] for d in INJ]),
     "inj_nd_avg": np.mean([B["NeighborDiv"][d][0] for d in INJ]), "arc_means": list(m)}
json.dump(K, open(OUT + "key.json", "w"), indent=1); print(json.dumps(K, indent=1))
