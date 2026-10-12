import sys, os, glob, json, csv, numpy as np
sys.path.insert(0, "/home/claude/exp/dirrule"); sys.path.insert(0, "/home/claude/exp/runs")
PKG = "/home/claude/exp/code/CoRe-GAD-MSDA-Reproducibility"; sys.path.insert(0, PKG + "/core_multiscale_directional")
import run_experiment as R, rules
from sklearn.metrics import roc_auc_score, average_precision_score
C = "/home/claude/exp/dirrule/cache"; SEALED = "/home/claude/exp/dirrule/sealed_labels"
SPLITS = {
 "dev": {"syn:cora_dev": "syn_dev_cora_*", "syn:citeseer_dev": "syn_dev_citeseer_*", "Cora": "Cora__*", "ACM": "ACM__*",
         "BlogCatalog": "BlogCatalog__*", "Amazon": "Amazon__*", "Amazon-all": "Amazon-all__*", "Tolokers": "Tolokers__*"},
 "t1": {**{n: f"{n}__*" for n in ["CiteSeer", "PubMed", "Flickr", "Facebook", "Reddit", "Disney", "Weibo", "YelpChi", "YelpChi-all", "Questions"]},
        "syn:cora_hold": "syn_hold_cora_*", "syn:citeseer_hold": "syn_hold_citeseer_*", "syn:pubmed_transfer": "syn_transfer_pubmed_*"},
 "t2": {n: f"{n}__*" for n in ["CS", "Photo", "Books", "Enron"]},
}
def labels_for(path, split):
    tag = os.path.basename(path)[:-4]
    if tag.startswith("syn_"): return np.load(f"{C}/labels_{tag}.npy")
    name = tag.split("__")[0]
    if split == "t2":
        assert os.path.exists("/home/claude/exp/dirrule/freeze.json"), "T2 labels are sealed until the rule is frozen"
        return np.load(f"{SEALED}/labels_{name}.npy")
    return np.load(f"{C}/labels_{name}.npy")
def run(split, cand_names=None):
    cands = rules.candidates(); cands = {k: v for k, v in cands.items() if cand_names is None or k in cand_names}
    rows = []
    for unit, pat in SPLITS[split].items():
        files = sorted(glob.glob(f"{C}/{pat}.npz"))
        assert files, (unit, pat)
        for f in files:
            z = np.load(f); y = labels_for(f, split)
            st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
            for cname, spec in cands.items():
                tails = rules.directions(spec, st, rk, mk)
                s = R.score_bank(st, rk, mk, ct, tuple(range(6)), (6, 7, 8), tails)
                au, ap, pi = roc_auc_score(y, s), average_precision_score(y, s), y.mean()
                rows.append(dict(split=split, unit=unit, run=os.path.basename(f)[:-4], cand=cname, auroc=au, ap=ap, prev=pi,
                                 U=0.5 * ((au - 0.5) / 0.5 + (ap - pi) / (1 - pi)), tails="".join(t[0] for t in tails)))
    return rows
def summarize(rows):
    out = {}
    for c in sorted({r["cand"] for r in rows}):
        units = sorted({r["unit"] for r in rows if r["cand"] == c})
        per = {u: {k: float(np.mean([r[k] for r in rows if r["cand"] == c and r["unit"] == u])) for k in ("U", "auroc", "ap")} for u in units}
        out[c] = dict(score=float(np.mean([p["U"] for p in per.values()])), auroc=float(np.mean([p["auroc"] for p in per.values()])),
                      ap=float(np.mean([p["ap"] for p in per.values()])), per_unit=per)
    return out
if __name__ == "__main__":
    split = sys.argv[1]; rows = run(split)
    with open(f"/home/claude/exp/dirrule/results_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    S = summarize(rows); json.dump(S, open(f"/home/claude/exp/dirrule/summary_{split}.json", "w"), indent=1)
    for c, v in sorted(S.items(), key=lambda kv: -kv[1]["score"]):
        print(f"{c:24s} U={v['score']:.4f} AUROC={v['auroc']:.4f} AP={v['ap']:.4f} | " + " ".join(f"{u[:9]}={p['auroc']:.3f}" for u, p in v["per_unit"].items()))
