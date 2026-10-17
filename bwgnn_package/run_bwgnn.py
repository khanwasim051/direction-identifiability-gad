"""Official BWGNN (squareRoot3/Rethinking-Anomaly-Detection @ de0631f) under the few-shot protocol
(BASELINES2_PROTOCOL.md, sha256 d548b49f...). Runs on the CPU, as the official code does.

    python run_bwgnn.py check          # short test on Books with one seed (seconds)
    python run_bwgnn.py dev            # the 16 development graphs
    python run_bwgnn.py heldout        # the 4 held-out graphs
Safe to stop and restart (finished parts are kept). Results are written to results/bwgnn_<graph>.csv; copy them to
results/official_dev/bwgnn/ or results/heldout/bwgnn/ of the repository. The graph files in data/ are produced by
make_bwgnn_data.py.
"""
import sys, os, csv, time, random, hashlib, urllib.request
import numpy as np, scipy.sparse as sp, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
URL = ("https://raw.githubusercontent.com/squareRoot3/Rethinking-Anomaly-Detection/"
       "de0631f039bbd19c1890b483cc01f1007f596af7/BWGNN.py")
SHA = "0ac45b1104f80fa17c50cd5e3bf9aa6241e008e6b57fdab9366ec69f80cb2f84"
DEV = ["Cora", "CiteSeer", "Facebook", "Disney", "Amazon", "Reddit", "YelpChi", "Tolokers", "Weibo-ARC",
       "BlogCatalog", "Flickr", "ACM", "PubMed", "Amazon-all", "Questions", "YelpChi-all"]
HELDOUT = ["Books", "Enron", "Photo", "CS"]
SEEDS = (11, 23, 37, 53, 71); KS = (1, 3, 5, 10)


def official_model():
    """Download the official BWGNN.py once, check that it is byte-identical to the pinned commit, import it."""
    path = os.path.join(HERE, "BWGNN.py")
    if not os.path.exists(path):
        print("Downloading the official BWGNN.py ...")
        urllib.request.urlretrieve(URL, path)
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != SHA:
        sys.exit("BWGNN.py does not match the official file. Delete it and run again.")
    sys.path.insert(0, HERE)
    import BWGNN
    return BWGNN.BWGNN


def load(name):
    z = np.load(os.path.join(DATA, f"{name}.npz"))
    if int(z["sparse"]):
        x = sp.csr_matrix((z["x_data"], z["x_indices"], z["x_indptr"]), shape=tuple(z["x_shape"])).toarray()
    else:
        x = z["x"]
    return x.astype(np.float32, copy=False), z["row"].astype(np.int64), z["col"].astype(np.int64), z["y"].astype(int)


def shots_for(y, seed, k):
    g = np.random.default_rng(np.random.SeedSequence([seed, 11000 + k]))
    return np.sort(g.choice(np.flatnonzero(y == 1), k, replace=False))


def val_split(y, shots, seed, k):
    g = np.random.default_rng(np.random.SeedSequence([seed, 12000 + k]))
    rest = np.setdiff1d(np.arange(len(y)), shots); out = []
    for c in (0, 1):
        idx = rest[y[rest] == c]; out.append(g.choice(idx, max(1, int(round(0.2 * len(idx)))), replace=False))
    return np.sort(np.concatenate(out))


def get_best_f1(labels, probs):          # copied from the official main.py
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels); preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1: best_f1, best_thre = mf1, thres
    return best_f1, best_thre


def run_one(BW, graph, feat, y, shots, val_idx, seed):
    """Official training loop (main.py) with training labels = shots (1) and all other nodes (0).
    One training run gives both variants: (a) scores of the last epoch, (b) scores of the epoch with the best
    validation macro-F1 (official selection rule)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = BW(feat.shape[1], 64, 2, graph, d=2)
    labels = torch.zeros(len(y), dtype=torch.long); labels[torch.tensor(shots)] = 1
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    weight = (1 - labels).sum().item() / labels.sum().item()
    best_f1, best_scores, last_scores = -1.0, None, None
    yv = y[val_idx]
    for e in range(100):
        model.train()
        logits = model(feat)
        loss = F.cross_entropy(logits, labels, weight=torch.tensor([1., weight]))
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        probs = logits.softmax(1).detach().numpy()
        f1, _ = get_best_f1(yv, probs[val_idx])
        if best_f1 < f1: best_f1, best_scores = f1, probs[:, 1].copy()
        last_scores = probs[:, 1]
    return last_scores, best_scores


def metrics(y, s, shots):
    rest = np.setdiff1d(np.arange(len(y)), shots)
    return roc_auc_score(y[rest], s[rest]), average_precision_score(y[rest], s[rest])


def run_graph(BW, name, seeds=SEEDS, ks=KS, tag=""):
    import dgl
    path = os.path.join(OUT, f"bwgnn_{name}{tag}.csv"); part = path + ".part"
    if os.path.exists(path): print(name, "already done"); return
    x, row, col, y = load(name)
    graph = dgl.graph((np.r_[row, col], np.r_[col, row]), num_nodes=len(y))
    feat = torch.from_numpy(x); del x  # no extra copy (saves memory)
    rows = list(csv.DictReader(open(part))) if os.path.exists(part) else []
    done = {int(r["seed"]) for r in rows}
    t0 = time.time()
    for seed in seeds:
        if seed in done: continue
        for k in ks:
            if k > int(y.sum()) // 2: continue
            sh = shots_for(y, seed, k); vi = val_split(y, sh, seed, k)
            last, best = run_one(BW, graph, feat, y, sh, vi, seed)
            for var, s in (("BWGNN_official_last", last), ("BWGNN_official_valsel", best)):
                au, ap = metrics(y, s, sh)
                rows.append(dict(dataset=name, seed=seed, k=k, method=var, auroc=au, ap=ap))
        with open(part, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"  {name}: seed {seed} done ({(time.time() - t0) / 60:.1f} min so far)", flush=True)
    os.replace(part, path)
    print(name, "finished", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    BW = official_model()
    import dgl
    print(f"torch {torch.__version__}, dgl {dgl.__version__}, threads {torch.get_num_threads()}")
    if what == "check":
        t = time.time(); run_graph(BW, "Books", seeds=(11,), ks=(1,), tag="_check")
        print(f"CHECK OK in {time.time() - t:.0f} s:", open(os.path.join(OUT, "bwgnn_Books_check.csv")).read())
    else:
        for name in (DEV if what == "dev" else HELDOUT): run_graph(BW, name)
        print("ALL DONE.")
