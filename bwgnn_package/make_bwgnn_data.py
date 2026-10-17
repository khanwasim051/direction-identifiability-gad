"""Exports the 20 graphs (features, undirected edges, labels) to bwgnn_package/data/<graph>.npz, the compact format
read by run_bwgnn.py. Run from the repository root after data/download_data.py:  python bwgnn_package/make_bwgnn_data.py
The package folder can then be copied to the machine that runs BWGNN (it needs only run_bwgnn.py, requirements.txt
and data/)."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import GADLIB, REPO  # noqa: E402
import sys, os, numpy as np, scipy.sparse as sp
sys.path.insert(0, str(GADLIB))
import bench_data as B

OUT = REPO / "bwgnn_package" / "data"; os.makedirs(OUT, exist_ok=True)
GRAPHS = ["Cora", "CiteSeer", "PubMed", "ACM", "BlogCatalog", "Flickr", "Facebook", "Reddit", "Amazon", "Disney",
          "Weibo-ARC", "Tolokers", "Amazon-all", "YelpChi-all", "Questions", "YelpChi", "CS", "Photo", "Books", "Enron"]
for n in (sys.argv[1:] or GRAPHS):
    x, a, y = B.load_any(n); a = sp.triu(sp.csr_matrix(a), 1).tocoo()
    common = dict(row=a.row.astype(np.int32), col=a.col.astype(np.int32), y=y.astype(np.int8))
    if (x != 0).mean() < 0.3:
        xs = sp.csr_matrix(x)
        np.savez_compressed(OUT / f"{n}.npz", x_data=xs.data, x_indices=xs.indices, x_indptr=xs.indptr,
                            x_shape=np.array(xs.shape), sparse=1, **common)
    else:
        np.savez_compressed(OUT / f"{n}.npz", x=x.astype(np.float32), sparse=0, **common)
    print(n, x.shape, int(y.sum()))
