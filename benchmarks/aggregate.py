import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import csv, glob, re, json, numpy as np, collections
R = str(RES_BENCH) + "/"
res = collections.defaultdict(dict)   # res[method][dataset] = (auroc_mean, auroc_sd, ap_mean, ap_sd, n)
def put(m, d, au, ap):
    au, ap = np.array(au, float), np.array(ap, float)
    res[m][d] = (au.mean(), au.std(ddof=1) if len(au) > 1 else 0.0, ap.mean(), ap.std(ddof=1) if len(ap) > 1 else 0.0, len(au))
# CoRe-GAD-MSDA and variants
for f in glob.glob(R + "core_out/metrics_*.csv"):
    rows = list(csv.DictReader(open(f)))
    for v in {r["variant"] for r in rows}:
        sel = [r for r in rows if r["variant"] == v]
        put({"directional_h1": "CoRe-GAD-MSDA"}.get(v, "CoRe:" + v), sel[0]["dataset"], [r["auroc"] for r in sel], [r["ap"] for r in sel])
for f in glob.glob(R + "tf_out/metrics_*.csv"):
    for r in csv.DictReader(open(f)): put(r["method"], r["dataset"], [r["auroc"]], [r["ap"]])
u = list(csv.DictReader(open(R + "unp_out_main.csv")))
for d in {r["dataset"] for r in u}:
    sel = [r for r in u if r["dataset"] == d]; put("UNPrompt (rerun)", d, [r["auroc"] for r in sel], [r["ap"] for r in sel])
ag = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(R + "anomalygfm_logs/*.log"):
    t = open(f).read()
    if "Epoch 300" not in t: continue
    blk = t.split("Epoch 300")[1]
    name = re.search(r"Testing (\S+)_svd AUC", blk).group(1)
    g = lambda pat: float(re.search(pat, blk).group(1))
    ag[name]["ab_au"].append(g(r"Abnormal Testing \S+ AUC_measure:([\d.]+)")); ag[name]["ab_ap"].append(g(r"Abnormal Testing AP_measure: ([\d.e-]+)"))
    ag[name]["lg_au"].append(g(r"Testing \S+ AUC:([\d.]+)")); ag[name]["lg_ap"].append(g(r"Testing AP: ([\d.e-]+)"))
for d, v in ag.items():
    put("AnomalyGFM (rerun)", d, v["ab_au"], v["ab_ap"]); put("AnomalyGFM (rerun, logit head)", d, v["lg_au"], v["lg_ap"])
# published (UNPrompt Table 1, AnomalyGFM Table 2, TAM Table 1)
PUB = {
 "UNPrompt (published)": {"Amazon": (.7525,.1602), "Reddit": (.5337,.0351), "Weibo": (.8860,.6406), "YelpChi": (.5875,.0712), "Amazon-all": (.7962,.2430), "YelpChi-all": (.5558,.1810), "Disney": (.6412,.1236), "Tolokers": (.6853,.2219), "Questions": (.4757,.0348)},
 "AnomalyGFM (published)": {"Reddit": (.5974,.0387), "Amazon": (.8417,.5790), "Disney": (.6751,.1242), "Amazon-all": (.9032,.6820), "YelpChi-all": (.5791,.1819), "Tolokers": (.5843,.2749), "Questions": (.5280,.0397)},
 "DOMINANT (published, per-graph)": {"BlogCatalog": (.7590,.3102), "ACM": (.8569,.4402), "Amazon": (.5996,.1424), "Facebook": (.5677,.0314), "Reddit": (.5555,.0356), "YelpChi": (.4133,.0395)},
 "CoLA (published, per-graph)": {"BlogCatalog": (.7746,.3270), "ACM": (.8233,.3235), "Amazon": (.5898,.0677), "Facebook": (.8434,.2106), "Reddit": (.6028,.0449), "YelpChi": (.4636,.0448)},
 "TAM (published, per-graph)": {"BlogCatalog": (.8248,.4182), "ACM": (.8878,.5124), "Amazon": (.7064,.2634), "Facebook": (.9144,.2233), "Reddit": (.6023,.0446), "YelpChi": (.5643,.0778)},
}
for m, dd in PUB.items():
    for d, (a, p) in dd.items(): res[m][d] = (a, None, p, None, None)
json.dump({m: {d: v for d, v in dd.items()} for m, dd in res.items()}, open(R + "all_results.json", "w"), indent=1)
ZS = ["Reddit", "Amazon", "Disney", "Weibo", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi"]
INJ = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook"]
def show(ds, methods, k):
    print("method".ljust(34) + "".join(d[:11].rjust(12) for d in ds))
    for m in methods:
        print(m[:33].ljust(34) + "".join((f"{res[m][d][k]:.4f}" if d in res[m] else "-").rjust(12) for d in ds))
M1 = ["NeighborDiv", "FreeGAD", "CoRe-GAD-MSDA", "CoRe:twosided_h1", "CoRe:h1_no_triangle", "CoRe:directional_h123", "UNPrompt (rerun)", "UNPrompt (published)", "AnomalyGFM (rerun)", "AnomalyGFM (rerun, logit head)", "AnomalyGFM (published)", "TAM (published, per-graph)"]
for k, lab in ((0, "AUROC"), (2, "AP")):
    print(f"\n=== Zero-shot organic suite {lab}"); show(ZS, M1, k)
M2 = ["NeighborDiv", "FreeGAD", "CoRe-GAD-MSDA", "CoRe:twosided_h1", "CoRe:h1_no_triangle", "DOMINANT (published, per-graph)", "CoLA (published, per-graph)", "TAM (published, per-graph)"]
for k, lab in ((0, "AUROC"), (2, "AP")):
    print(f"\n=== Injected suite {lab}"); show(INJ, M2, k)
