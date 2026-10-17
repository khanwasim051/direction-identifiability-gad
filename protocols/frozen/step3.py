"""Executes STEP3_PROTOCOL.md (sha256 in STEP3_PROTOCOL.sha256). Output: results3/step3.csv"""
import sys, csv, numpy as np
sys.path.insert(0, "/home/claude/exp/fsdc")
import fsdc as F, step2 as S2


def final_scores(name_stats, st, rk, mk, ct, ga, sh, k):
    """Returns dict of scores V0, V2, W1, W2 for one graph/seed/shot set."""
    grel = F.R.marginal_ranks(st)
    gr = np.c_[grel, ga]; m = np.c_[mk, np.ones_like(ga)]
    frozen = list(F.FROZEN) + ["upper"] * ga.shape[1]
    t9, s9 = F.calibrate(F.shot_aucs(grel, mk, sh))
    v0 = F.score(st, rk, mk, ct, t9, s9)
    tails, s = S2.calibrate(gr, m, sh, frozen)
    tau = 1 / np.sqrt(12 * k); w2 = np.maximum(0, s - tau)
    if w2.sum() == 0: w2 = s
    v2 = S2.pooled(gr, m, tails, w2)
    wa = np.maximum(0, s - tau); wa[:9] = 0                     # attribute statistics only
    att = S2.pooled(gr, m, tails, wa) if wa.sum() > 0 else None
    w1 = v2 if k >= 3 else v0
    w2s = np.maximum(v0, att) if att is not None else v0
    return {"V0_M2": v0, "V2_shrink": v2, "W1_kgated": w1, "W2_maxfusion": w2s}


def main():
    rows = []
    for name in F.GRAPHS:
        ga, y = S2.attribute_stats(name)
        for seed in F.SEEDS:
            z = np.load(f"{F.CACHE}/{name}__{seed}.npz")
            for k in F.cells(y):
                sh = F.shots_for(y, seed, k)
                for meth, sc in final_scores(name, z["stats"], z["ranks"], z["mask"], z["counts"], ga, sh, k).items():
                    au, ap = F.metrics(y, sc, sh)
                    rows.append(dict(dataset=name, seed=seed, k=k, method=meth, auroc=au, ap=ap,
                                     group="injected" if name in S2.INJECTED else "organic"))
        print("done", name, flush=True)
    with open(f"{S2.OUT}/step3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
