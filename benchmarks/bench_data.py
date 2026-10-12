"""Unified loaders for the standard GAD benchmark .mat files (labels returned separately, used only for metrics)."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import numpy as np, scipy.io as sio, scipy.sparse as sp
from pathlib import Path
D = Path(DATA)
ARC = D / "yixinliu233_ARC/dataset"
FILES = {
    # zero-shot generalist suite (UNPrompt / AnomalyGFM targets)
    "Reddit": (ARC / "Reddit.mat", "Network", "Attributes", "Label"),
    "Amazon": (ARC / "Amazon.mat", "Network", "Attributes", "Label"),
    "Disney": (D / "mala-lab_UNPrompt/Datasets/Disney.mat", "Network", "Attributes", "Label"),
    "Amazon-all": (D / "raw/Amazon.mat", "homo", "features", "label"),
    "YelpChi-all": (D / "raw/YelpChi.mat", "homo", "features", "label"),
    "Tolokers": (ARC / "tolokers.mat", "Network", "Attributes", "Label"),
    "Questions": (ARC / "questions.mat", "Network", "Attributes", "Label"),
    "Weibo": (D / "raw/unprompt_weibo/weibo.mat", "Network", "Attributes", "Label"),
    "Weibo-ARC": (ARC / "weibo.mat", "Network", "Attributes", "Label"),
    "YelpChi": (ARC / "YelpChi.mat", "Network", "Attributes", "Label"),
    # injected-anomaly suite (ARC / CoLA protocol)
    "Cora": (ARC / "cora.mat", "Network", "Attributes", "Label"),
    "CiteSeer": (ARC / "citeseer.mat", "Network", "Attributes", "Label"),
    "PubMed": (ARC / "pubmed.mat", "Network", "Attributes", "Label"),
    "ACM": (ARC / "ACM.mat", "Network", "Attributes", "Label"),
    "BlogCatalog": (ARC / "BlogCatalog.mat", "Network", "Attributes", "Label"),
    "Flickr": (ARC / "Flickr.mat", "Network", "Attributes", "Label"),
    "Facebook": (ARC / "Facebook.mat", "Network", "Attributes", "Label"),
}

def clean(a):
    a = sp.csr_matrix(a, dtype=np.float32)
    a = a.maximum(a.T).tocsr(); a.setdiag(0); a.eliminate_zeros()
    a.data[:] = 1; a = a.astype(np.int8); a.sort_indices()
    return a

def load(name):
    path, ak, xk, yk = FILES[name]
    d = sio.loadmat(path)
    x = d[xk]; x = np.asarray(x.todense() if sp.issparse(x) else x, dtype=np.float32)
    y = (np.asarray(d[yk]).ravel() != 0).astype(np.int8)
    return x, clean(d[ak]), y

# --- T2 (untouched) datasets, added for the direction-rule study ---
FILES["CS"] = (ARC / "cs.mat", "Network", "Attributes", "Label")
FILES["Photo"] = (ARC / "photo.mat", "Network", "Attributes", "Label")
PYGOD = {"Books": D / "pygod/books.pt", "Enron": D / "pygod/enron.pt"}

def load_any(name):
    if name in PYGOD:
        import safe_pt
        x, e, y = safe_pt.load(str(PYGOD[name]))
        n = x.shape[0]
        a = sp.csr_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n))
        return np.asarray(x, dtype=np.float32), clean(a), (np.asarray(y).ravel() != 0).astype(np.int8)
    return load(name)
