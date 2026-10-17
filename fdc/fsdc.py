"""Executes FEASIBILITY_PROTOCOL.md (sha256 in FEASIBILITY_PROTOCOL.sha256).
Usage: python fsdc.py ours            -> results/ours.csv  (M0, M1, M2 on all graphs)
       python fsdc.py base GRAPH ...  -> results/base_GRAPH.csv (B1 GDN, B2 BWGNN)
"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import PKG, GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES_R1, RES_DEV, RES_R23, RES_HO, RES_TAB  # noqa: E402
import sys, os, csv, time, gc, numpy as np
sys.path.insert(0, str(GADLIB))
sys.path.insert(0, str(PKG / "core_multiscale_directional"))
import run_experiment as R
from sklearn.metrics import roc_auc_score, average_precision_score

CACHE = str(CACHE)
OUT = str(RES_R1); os.makedirs(OUT, exist_ok=True)
GRAPHS = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook",
          "Reddit", "Amazon", "Disney", "Weibo-ARC", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi"]
SEEDS = (11, 23, 37, 53, 71); KS = (1, 3, 5, 10)
FROZEN = ("upper", "upper", "upper", "lower", "two-sided", "upper", "lower", "upper", "upper")
LOCAL, STRUCT = tuple(range(6)), (6, 7, 8)


def shots_for(y, seed, k):
    g = np.random.default_rng(np.random.SeedSequence([seed, 11000 + k]))
    return np.sort(g.choice(np.flatnonzero(y == 1), k, replace=False))


def cells(y):
    na = int(y.sum())
    return [k for k in KS if k <= na // 2]


def metrics(y, s, shots):
    rest = np.setdiff1d(np.arange(len(y)), shots)
    return roc_auc_score(y[rest], s[rest]), average_precision_score(y[rest], s[rest])


# ---------------- CoRe-GAD-MSDA with calibrated directions ----------------
def shot_aucs(gr, mk, shots):
    """Per statistic: (AUC_up, AUC_low, AUC_two) of the anomaly shots against the graph; None if no valid shot."""
    out = []
    for j in range(gr.shape[1]):
        a = shots[mk[shots, j] > 0]
        if len(a) == 0: out.append(None); continue
        up = float(np.mean((gr[a, j] + 1) / 2)); out.append({"upper": up, "lower": 1 - up, "two-sided": float(np.mean(np.abs(gr[a, j])))})
    return out


def calibrate(aucs):
    tails, sep = [], []
    for j, d in enumerate(aucs):
        if d is None: tails.append(FROZEN[j]); sep.append(0.0); continue
        best = max(d.values())
        cands = [t for t in ("upper", "lower", "two-sided") if abs(d[t] - best) <= 1e-12]
        t = FROZEN[j] if FROZEN[j] in cands else cands[0]
        tails.append(t); sep.append(max(0.0, d[t] - 0.5))
    return tuple(tails), np.array(sep)


def score(st, rk, mk, ct, tails, mult=None):
    """R.score_bank with an optional per-statistic weight multiplier (M2)."""
    ev = R.orient(rk, mk, tails)
    vc = mk.sum(axis=0)
    support = np.clip(ct / np.maximum(vc - 1, 1)[None, :], 0, 1) * mk
    disp = np.zeros(st.shape[1])
    for j in range(st.shape[1]):
        v = mk[:, j] > 0
        if v.any():
            q1, q3 = np.quantile(ev[v, j], (0.25, 0.75)); disp[j] = np.clip((q3 - q1) / 0.5, 0, 1)
    stab = np.clip(1 - np.abs(rk - R.marginal_ranks(st)) / 2, 0, 1) * mk
    w = support * disp[None, :] * stab
    if mult is not None: w = w * mult[None, :]
    return np.maximum(R.masked_mean(ev[:, LOCAL], w[:, LOCAL]), R.masked_mean(ev[:, STRUCT], w[:, STRUCT]))


def run_ours():
    rows = []
    for name in GRAPHS:
        y = np.load(f"{CACHE}/labels_{name}.npy").astype(int)
        for seed in SEEDS:
            z = np.load(f"{CACHE}/{name}__{seed}.npz"); st, rk, mk, ct = z["stats"], z["ranks"], z["mask"], z["counts"]
            gr = R.marginal_ranks(st)
            # sanity: frozen pipeline must equal the original detector
            assert np.allclose(score(st, rk, mk, ct, FROZEN), R.score_bank(st, rk, mk, ct, LOCAL, STRUCT, FROZEN))
            s0 = score(st, rk, mk, ct, FROZEN)
            for k in cells(y):
                sh = shots_for(y, seed, k)
                tails, sep = calibrate(shot_aucs(gr, mk, sh))
                for meth, s in (("M0_frozen", s0), ("M1_DIR", score(st, rk, mk, ct, tails)),
                                ("M2_DIR-W", score(st, rk, mk, ct, tails, sep))):
                    au, ap = metrics(y, s, sh)
                    rows.append(dict(dataset=name, seed=seed, k=k, method=meth, auroc=au, ap=ap,
                                     tails="".join(t[0] for t in tails), sep=" ".join(f"{v:.3f}" for v in sep)))
        print("ours done", name, flush=True)
    with open(f"{OUT}/ours.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ---------------- baselines (re-implemented, fixed hyperparameters) ----------------
def graph_tensors(name):
    import torch, scipy.sparse as sp, bench_data as B
    x, a, y = B.load_any(name)
    x = x / np.maximum(np.abs(x).sum(1, keepdims=True), 1e-12)          # row-normalise
    a = sp.csr_matrix(a, dtype=np.float32); n = a.shape[0]
    def to_t(m):
        m = m.tocoo(); return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data.astype(np.float32), m.shape).coalesce()
    ah = a + sp.eye(n, dtype=np.float32); d = np.asarray(ah.sum(1)).ravel()
    gcn = sp.diags(d ** -0.5) @ ah @ sp.diags(d ** -0.5)                  # GCN propagation
    deg = np.asarray(a.sum(1)).ravel(); dinv = np.where(deg > 0, deg ** -0.5, 0)
    lap = sp.eye(n, dtype=np.float32) - sp.diags(dinv) @ a @ sp.diags(dinv)  # normalised Laplacian
    return torch.tensor(x), to_t(gcn), to_t(lap), y.astype(int)


def symmm(M):
    """M @ h for a symmetric sparse M, with gradient M @ g (exact; faster than torch.sparse.mm backward)."""
    import torch
    Mc = M.to_sparse_csr()
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h): return Mc @ h
        @staticmethod
        def backward(ctx, g): return Mc @ g
    return F.apply


def spdense(X, thr=0.3):
    """W -> X @ W with gradient X^T @ g; uses a sparse X when its density is below thr (exact up to rounding)."""
    import torch
    if (X != 0).float().mean().item() >= thr: return lambda W: X @ W
    Xs, Xt = X.to_sparse_csr(), X.T.contiguous().to_sparse_csr()
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, W): return Xs @ W
        @staticmethod
        def backward(ctx, g): return Xt @ g
    return F.apply


def beta_thetas(d=2):
    from scipy.special import beta
    th = []
    for i in range(d + 1):
        p = np.poly1d([0.5, 0]) ** i * np.poly1d([-0.5, 1]) ** (d - i) / beta(i + 1, d + 1 - i)
        th.append(p.coeffs[::-1].tolist() + [0.0] * (d + 1 - len(p.coeffs)))  # ascending powers of L
    return th


def run_gdn(x, P, y, shots, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    class GDN(nn.Module):
        def __init__(s, f, h=64):
            super().__init__(); s.l1, s.l2, s.out = nn.Linear(f, h), nn.Linear(h, h), nn.Linear(h, 1)
        def forward(s, x):
            h = torch.relu(Px(s.l1.weight.T) + P1 * s.l1.bias); h = torch.relu(mm(s.l2(h)))
            return s.out(h).squeeze(-1)
    mm = symmm(P); Px = spdense(mm(x)); P1 = mm(torch.ones(len(y), 1))
    m = GDN(x.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-4)
    unl = np.setdiff1d(np.arange(len(y)), shots); ref = torch.randn(5000)
    for ep in range(100):
        for step in range(10):
            b = g.choice(unl, min(512, len(unl)), replace=False)
            idx = torch.tensor(np.r_[shots, b]); lab = torch.tensor(np.r_[np.ones(len(shots)), np.zeros(len(b))], dtype=torch.float32)
            dev = (m(x)[idx] - ref.mean()) / ref.std()
            loss = ((1 - lab) * dev.abs() + lab * torch.clamp(5.0 - dev, min=0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(x).numpy()


def run_bwgnn(x, L, y, shots, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); th = beta_thetas(2); mm = symmm(L); X1 = spdense(x)
    class BW(nn.Module):
        def __init__(s, f, h=64):
            super().__init__(); s.l1, s.l2 = nn.Linear(f, h), nn.Linear(h, h)
            s.l3, s.l4 = nn.Linear(h * len(th), h), nn.Linear(h, 2)
        def forward(s, x):
            h = torch.relu(s.l2(torch.relu(X1(s.l1.weight.T) + s.l1.bias)))
            pw = [h]; [pw.append(mm(pw[-1])) for _ in range(len(th[0]) - 1)]
            hs = [sum(c * p for c, p in zip(t, pw)) for t in th]
            return s.l4(torch.relu(s.l3(torch.cat(hs, 1))))
    m = BW(x.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=0.01)
    lab = torch.zeros(len(y), dtype=torch.long); lab[torch.tensor(shots)] = 1
    wpos = (len(y) - len(shots)) / len(shots); ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, wpos]))
    for ep in range(100):
        loss = ce(m(x), lab); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return torch.softmax(m(x), 1)[:, 1].numpy()


def run_base(names):
    import torch; torch.set_num_threads(1)
    for name in names:
        path = f"{OUT}/base_{name}.csv"
        if os.path.exists(path): continue
        t0 = time.time(); x, P, L, y = graph_tensors(name)
        ycache = np.load(f"{CACHE}/labels_{name}.npy").astype(int); assert (ycache == y).all()
        part = path + ".part"; rows = list(csv.DictReader(open(part))) if os.path.exists(part) else []
        done = {int(r["seed"]) for r in rows}
        for seed in SEEDS:
            if seed in done: continue
            for k in cells(y):
                sh = shots_for(y, seed, k)
                for meth, fn, M in (("B1_GDN", run_gdn, P), ("B2_BWGNN", run_bwgnn, L)):
                    s = fn(x, M, y, sh, seed); au, ap = metrics(y, s, sh); gc.collect()  # free per-run autograd classes
                    rows.append(dict(dataset=name, seed=seed, k=k, method=meth, auroc=au, ap=ap))
            with open(part, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            print(name, seed, f"{time.time() - t0:.0f}s", flush=True)
        with open(path + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    if sys.argv[1] == "ours": run_ours()
    else: run_base(sys.argv[2:])
