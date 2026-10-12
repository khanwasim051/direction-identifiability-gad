"""One-process efficiency benchmark for one method and one citation graph."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "baseline_comparison"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE / "condcal_citation_exact_global"))
sys.path.insert(0, str(WORKSPACE / "condcal_cora"))

from condcal import Config, GRAPH_IDS, extract, load_planetoid, preprocess, rng_for  # noqa: E402
from run_exact_global_citation import exact_marginal  # noqa: E402
from run_four_mechanism_stress import (  # noqa: E402
    ANOMALY_FRACTION, INJECTORS, structural_statistics,
)
from run_neighbor_div import neighbor_div_scores, projected_features  # noqa: E402
from run_freegad import freegad_scores, row_normalize_features  # noqa: E402
from run_reliability_fusion import score_arrays  # noqa: E402


def graph_data(graph):
    raw, base = load_planetoid(graph, WORKSPACE / "condcal_cora" / "data")
    normalized, _, _ = preprocess(raw)
    adjacency, _, _, _ = INJECTORS["cross_community_edges"](
        base,
        normalized,
        int(np.floor(ANOMALY_FRACTION * len(raw))),
        rng_for(11, GRAPH_IDS[graph], 100),
    )
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return raw, normalized, adjacency


def core_score(normalized, adjacency, graph):
    local_statistics, _, _ = extract(
        normalized, adjacency, Config(), rng_for(11, GRAPH_IDS[graph], 200)
    )
    local_rank, local_mask, local_counts, _ = exact_marginal(
        local_statistics, adjacency
    )
    structural_stats, _ = structural_statistics(normalized, adjacency)
    structural_rank, structural_mask, structural_counts, _ = exact_marginal(
        structural_stats, adjacency
    )
    arrays = {
        "local_statistics": local_statistics,
        "local_rank": local_rank,
        "local_mask": local_mask,
        "local_counts": local_counts,
        "structural_statistics": structural_stats,
        "structural_rank": structural_rank,
        "structural_mask": structural_mask,
        "structural_counts": structural_counts,
    }
    scores, _, _ = score_arrays(arrays)
    return scores["reliability_branch_max"], 0


def dominant_score(raw, adjacency, graph):
    import torch
    from pygod.nn import DOMINANTBase
    from run_dominant_exact import fit_score

    torch.set_num_threads(8)
    scores, _ = fit_score(raw.astype(np.float32), adjacency, 11)
    model = DOMINANTBase(
        in_dim=raw.shape[1], hid_dim=64, num_layers=4,
        dropout=0.0, sigmoid_s=False,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return scores, parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("core", "neighbor_div", "freegad", "dominant"))
    parser.add_argument("graph", choices=("cora", "citeseer", "pubmed"))
    args = parser.parse_args()
    started = time.perf_counter()
    raw, normalized, adjacency = graph_data(args.graph)
    if args.method == "core":
        scores, parameters = core_score(normalized, adjacency, args.graph)
        training = False
        epochs = 0
    elif args.method == "neighbor_div":
        scores, _, _, _ = neighbor_div_scores(projected_features(raw), adjacency)
        parameters = 0
        training = False
        epochs = 0
    elif args.method == "freegad":
        scores, _, _, _ = freegad_scores(row_normalize_features(raw), adjacency)
        parameters = 0
        training = False
        epochs = 0
    else:
        scores, parameters = dominant_score(raw, adjacency, args.graph)
        training = True
        epochs = 20
    result = {
        "method": args.method,
        "graph": args.graph,
        "representative_mechanism": "cross_community_edges",
        "seed": 11,
        "seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "trainable_parameters": int(parameters),
        "training_required": training,
        "epochs": epochs,
        "nodes": len(scores),
        "finite_scores": bool(np.isfinite(scores).all()),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
