"""Revision: official GDN (kaize0409/Meta-GDN_AnomalyDetection @426516b) exactly as in BASELINES2_PROTOCOL.md, with 20 seeds
(11, 23, 37, 53, 71, 101-115) and the extended metrics (AUROC, AP, nAP, R-precision).
Usage: python fdc/rev_gdn20.py GRAPH ...   -> results/revision/gdn20_GRAPH.csv"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, FDC, CACHE, SEALED, OFFICIAL, REPO, RES, RES_TAB  # noqa: E402
import sys, os, csv, time, random, numpy as np, torch
OFF = str(OFFICIAL / "Meta-GDN_AnomalyDetection"); sys.path.insert(0, OFF)
sys.path.insert(0, str(GADLIB)); sys.path.insert(0, str(FDC))
from models import SGC                                   # official
from meta import dev_loss                                # official
from utils import sgc_precompute, normalize_adjacency    # official
from data import sp_matrix_to_torch_sparse_tensor, test_task_generator_backup  # official
import fsdc as F, bench_data as B
from sklearn.metrics import roc_auc_score

OUT = str(RES / "revision"); os.makedirs(OUT, exist_ok=True)
SEALED = str(SEALED); HELD = {"CS", "Photo", "Books", "Enron"}
SEEDS = (11, 23, 37, 53, 71) + tuple(range(101, 116))
from rev_eval import metrics as MET


def val_split(y, shots, seed, k):
    """Stratified random 20% of the non-shot nodes (variant b only)."""
    g = np.random.default_rng(np.random.SeedSequence([seed, 12000 + k]))
    rest = np.setdiff1d(np.arange(len(y)), shots); out = []
    for c in (0, 1):
        idx = rest[y[rest] == c]; out.append(g.choice(idx, max(1, int(round(0.2 * len(idx)))), replace=False))
    return np.sort(np.concatenate(out))


def run(feature, y, shots, seed, val_idx=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    k = len(shots); unl = np.setdiff1d(np.arange(len(y)), shots).tolist()
    model = SGC(feature.shape[1], 1)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    best, best_state = -1.0, None
    for e in range(100):
        x_tr, y_tr, _, _ = test_task_generator_backup(feature, shots.tolist(), unl, 2 * k, y, [0], "cpu")
        model.train(); optim.zero_grad(); loss = dev_loss(y_tr, model(x_tr)); loss.backward(); optim.step()
        if val_idx is not None:
            model.eval()
            with torch.no_grad(): s = model(feature[val_idx]).numpy().ravel()
            a = roc_auc_score(y[val_idx], s)
            if a > best: best, best_state = a, {n: p.clone() for n, p in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): return model(feature).numpy().ravel()


def main(names):
    torch.set_num_threads(2)
    for name in names:
        path = f"{OUT}/gdn20_{name}.csv"
        if os.path.exists(path): continue
        x, a, y = B.load_any(name); y = y.astype(int)
        assert (np.load(f"{SEALED if name in HELD else F.CACHE}/labels_{name}.npy").astype(int) == y).all()
        adj = sp_matrix_to_torch_sparse_tensor(normalize_adjacency(a.astype(np.float64))).float()
        feature = sgc_precompute(torch.FloatTensor(x), adj, 2)
        rows, t0 = [], time.time()
        for seed in SEEDS:
            for k in F.cells(y):
                sh = F.shots_for(y, seed, k)
                for var, vi in (("GDN_official_last", None), ("GDN_official_valsel", val_split(y, sh, seed, k))):
                    rows.append(dict(dataset=name, seed=seed, k=k, method=var, **MET(y, run(feature, y, sh, seed, vi), sh)))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
