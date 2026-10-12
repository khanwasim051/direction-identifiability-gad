"""Prepare working copies of the benchmark files for the official UNPrompt and AnomalyGFM code.

UNPrompt reads ./Datasets/<name>.mat; AnomalyGFM reads ./data/<name>_svd.mat with features
projected to 8 dimensions by SVD, exactly as in AnomalyGFM's svd_feature_trans.py
(reduced SVD gives the same leading components and fits in memory)."""
import shutil
import numpy as np, scipy.io as sio, scipy.sparse as sp
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, BENCH  # noqa: E402
_sys.path.insert(0, str(BENCH))
import bench_data as B  # noqa: E402

UNP = WORK / "unp_work" / "Datasets"; AGFM = WORK / "agfm_work" / "data"
UNP.mkdir(parents=True, exist_ok=True); AGFM.mkdir(parents=True, exist_ok=True); (WORK / "unp_work" / "results").mkdir(exist_ok=True)
ORIGINAL = {  # files used as distributed
    "Facebook": DATA / "mala-lab_UNPrompt/Datasets/Facebook.mat", "Amazon": DATA / "mala-lab_UNPrompt/Datasets/Amazon.mat",
    "Reddit": DATA / "mala-lab_UNPrompt/Datasets/Reddit.mat", "Disney": DATA / "mala-lab_UNPrompt/Datasets/Disney.mat",
    "Weibo": DATA / "raw/unprompt_weibo/weibo.mat", "Tolokers": DATA / "yixinliu233_ARC/dataset/tolokers.mat",
    "Questions": DATA / "yixinliu233_ARC/dataset/questions.mat", "YelpChi": DATA / "yixinliu233_ARC/dataset/YelpChi.mat"}
for name, path in ORIGINAL.items():
    shutil.copy(path, UNP / f"{name}.mat")
for name in ("Amazon-all", "YelpChi-all"):   # CARE-GNN 'homo' relation, renamed keys only (graph symmetrised, self-loops removed)
    x, a, y = B.load(name)
    sio.savemat(UNP / f"{name}.mat", {"Network": sp.csc_matrix(a.astype(np.float64)), "Attributes": sp.csc_matrix(x.astype(np.float64)),
                                      "Label": y.reshape(1, -1).astype(np.float64)}, do_compression=True)
for name in ["Facebook", "Amazon", "Reddit", "Disney", "Weibo", "Tolokers", "Amazon-all"]:
    d = sio.loadmat(UNP / f"{name}.mat"); attr = d["Attributes"]
    feat = np.array(attr.todense() if sp.issparse(attr) else attr, dtype=np.float64)
    U, S, _ = np.linalg.svd(feat, full_matrices=False)
    sio.savemat(AGFM / f"{name}_svd.mat", {"Network": d["Network"], "Label": d["Label"], "Attributes": U[:, :8] @ np.diag(S[:8])})
    print("prepared", name, flush=True)
