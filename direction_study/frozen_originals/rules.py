"""Label-free direction rules (exactly as listed in PROTOCOL.md). Inputs: cached stats/ranks/mask of one graph."""
import numpy as np
from scipy.stats import skew, spearmanr
FROZEN = ("upper", "upper", "upper", "lower", "two-sided", "upper", "lower", "upper", "upper")
FLIP = {"upper": "lower", "lower": "upper"}

def skewness(v, kind):
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.ptp(v) == 0: return None
    if kind == "moment": return float(skew(v, bias=False))
    q10, q50, q90 = np.quantile(v, (0.1, 0.5, 0.9))
    return 0.0 if q90 == q10 else float((q90 + q10 - 2 * q50) / (q90 - q10))

def rule_A(stats, kind, tau):
    out = []
    for k in range(stats.shape[1]):
        s = skewness(stats[:, k], kind)
        out.append(FROZEN[k] if s is None else "upper" if s > tau else "lower" if s < -tau else "two-sided")
    return tuple(out)

def rule_B(stats, kind, tau):
    out = []
    for k in range(stats.shape[1]):
        s, d = skewness(stats[:, k], kind), FROZEN[k]
        if s is None or abs(s) <= tau: out.append(d); continue
        want = "upper" if s > 0 else "lower"
        out.append(want if d == "two-sided" else (d if d == want else FLIP[d]))
    return tuple(out)

def _evidence(r, m, tail):
    return ((r + 1) / 2 if tail == "upper" else (1 - r) / 2 if tail == "lower" else np.abs(r)) * m

def rule_consensus(ranks, mask, init):
    tails = list(init)
    for _ in range(10):
        changed = False
        for k in range(ranks.shape[1]):
            E = np.column_stack([_evidence(ranks[:, j], mask[:, j], tails[j]) for j in range(ranks.shape[1])])
            others = [j for j in range(ranks.shape[1]) if j != k]
            cnt = mask[:, others].sum(1); valid = (mask[:, k] > 0) & (cnt > 0)
            if valid.sum() < 3: continue
            ref = (E[:, others].sum(1)[valid] / cnt[valid])
            best, bc = tails[k], -np.inf
            for t in ("upper", "lower", "two-sided"):
                e = _evidence(ranks[valid, k], 1.0, t)
                c = spearmanr(e, ref).statistic if np.ptp(e) > 0 and np.ptp(ref) > 0 else -np.inf
                if np.isfinite(c) and c > bc + 1e-12: best, bc = t, c
            if best != tails[k]: tails[k] = best; changed = True
        if not changed: break
    return tuple(tails)

def candidates():
    c = {"F0_frozen": ("F0",), "F1_twosided": ("F1",)}
    for kind, taus in (("moment", (0, 0.5, 1.0)), ("kelly", (0, 0.1, 0.25))):
        for t in taus:
            c[f"A_{kind}_{t}"] = ("A", kind, t); c[f"B_{kind}_{t}"] = ("B", kind, t)
    c["C_consensus_frozen"] = ("C",); c["D_consensus_kelly0.1"] = ("D",)
    return c

def directions(spec, stats, ranks, mask):
    if spec[0] == "F0": return FROZEN
    if spec[0] == "F1": return ("two-sided",) * 9
    if spec[0] == "A": return rule_A(stats, spec[1], spec[2])
    if spec[0] == "B": return rule_B(stats, spec[1], spec[2])
    if spec[0] == "C": return rule_consensus(ranks, mask, FROZEN)
    if spec[0] == "D": return rule_consensus(ranks, mask, rule_A(stats, "kelly", 0.1))
