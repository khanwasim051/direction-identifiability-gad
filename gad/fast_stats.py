"""Exact vectorized equivalents of the per-node loops used by CoRe-GAD-MSDA and NeighborDiv.

structural_statistics_fast reproduces run_four_mechanism_stress.structural_statistics:
  col0 ego-triangle load  = triangles(i) / d_i   (0 when d_i < 2)
  col1 unexpected-edge tail = mean of the ceil(sqrt(d_i)) largest incident cosine dissimilarities
  col2 motif incoherence  = mean over triangles (i,u,v) of (3 - s_iu - s_iv - s_uv)/3 (NaN if none)
using the identities
  sum_tri (s_iu + s_iv) = sum_u A_iu S_iu C_iu,   C = (A A) o A   (common-neighbour counts on edges)
  sum_tri s_uv          = 1/2 sum_u A_iu (A S)_iu
computed in row blocks so memory stays bounded.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA  # noqa: E402
import numpy as np, scipy.sparse as sp

def _blocks(a, budget=4e7):
    d = np.diff(a.indptr).astype(np.float64)
    work = a @ d            # per-row cost of A[R] @ A
    start, acc = 0, 0.0
    for i in range(a.shape[0]):
        acc += work[i]
        if acc > budget:
            yield start, i + 1; start, acc = i + 1, 0.0
    if start < a.shape[0]:
        yield start, a.shape[0]

def edge_similarity(x, a):
    a = a.tocoo()
    s = np.empty(len(a.row), dtype=np.float64)
    step = max(1, int(2e7 // max(x.shape[1], 1)))
    for lo in range(0, len(a.row), step):
        r, c = a.row[lo:lo + step], a.col[lo:lo + step]
        s[lo:lo + step] = np.einsum("ij,ij->i", x[r], x[c])
    return sp.csr_matrix((s, (a.row, a.col)), shape=a.shape)

def structural_statistics_fast(features, adjacency):
    x = np.asarray(features, dtype=np.float32)
    a = sp.csr_matrix(adjacency, dtype=np.float64); a.sort_indices()
    n = a.shape[0]; d = np.diff(a.indptr)
    S = edge_similarity(x, a)
    tri = np.zeros(n); pair_sum = np.zeros(n); uv_sum = np.zeros(n)
    for lo, hi in _blocks(a):
        ar = a[lo:hi]
        C = (ar @ a).multiply(ar).tocsr()                      # common neighbours on edges
        tri[lo:hi] = np.asarray(C.sum(axis=1)).ravel() / 2.0
        pair_sum[lo:hi] = np.asarray(C.multiply(S[lo:hi]).sum(axis=1)).ravel()
        AS = (ar @ S).multiply(ar)
        uv_sum[lo:hi] = np.asarray(AS.sum(axis=1)).ravel() / 2.0
    out = np.full((n, 3), np.nan)
    out[:, 0] = 0.0
    m2 = d >= 2
    out[m2, 0] = tri[m2] / d[m2]
    has = m2 & (tri > 0)
    out[has, 2] = (3 * tri[has] - pair_sum[has] - uv_sum[has]) / (3 * tri[has])
    out[has, 2] = np.clip(out[has, 2], 0.0, 2.0)
    # unexpected-edge tail (loop is O(sum d))
    for i in np.flatnonzero(d > 0):
        nb = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
        inc = np.clip(1.0 - x[nb] @ x[i], 0.0, 2.0)
        k = max(1, int(np.ceil(np.sqrt(len(nb)))))
        out[i, 1] = float(np.mean(np.partition(inc, -k)[-k:]))
    return out, tri.astype(np.int64)

def neighbor_div_fast(embedding, adjacency):
    """Exact NeighborDiv: variance (ddof=0) of e_u.e_v over unordered neighbour pairs u<v,
    via sum_{u<v} s = (|sum e|^2 - sum|e|^2)/2 and sum_{u<v} s^2 = (|E^T E|_F^2 - sum |e|^4)/2."""
    e = np.asarray(embedding, dtype=np.float64)
    a = sp.csr_matrix(adjacency, dtype=np.float64)
    n = a.shape[0]; d = np.diff(a.indptr)
    r = e.shape[1]
    sq = (e * e).sum(1)
    s1 = a @ e                                                   # sum of neighbour embeddings
    outer = np.einsum("ij,ik->ijk", e, e).reshape(n, r * r)
    G = a @ outer                                                # sum e e^T over neighbours
    sum_sq = a @ sq; sum_q = a @ (sq ** 2)
    pairs = d * (d - 1) / 2.0
    tot = ((s1 ** 2).sum(1) - sum_sq) / 2.0
    tot2 = ((G ** 2).sum(1) - sum_q) / 2.0
    valid = d >= 2
    div = np.full(n, np.nan)
    mean = tot[valid] / pairs[valid]
    div[valid] = np.maximum(tot2[valid] / pairs[valid] - mean ** 2, 0.0)
    reference = np.median(div[valid]); dev = np.abs(div - reference)
    mu, sd = dev[valid].mean(), dev[valid].std(ddof=0)
    scores = np.zeros(n)
    if sd > 0: scores[valid] = (dev[valid] - mu) / sd
    return scores, div, valid, reference
