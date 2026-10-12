"""Fresh-process efficiency benchmark for the frozen evidence bank."""
import argparse, json, resource, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_experiment import (  # noqa: E402
    citation_instance, residual_statistics, score_bank, structural_statistics,
)
from condcal import Config, GRAPH_IDS, extract  # noqa: E402
from run_exact_global_citation import exact_marginal  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("graph", choices=("cora", "citeseer", "pubmed"))
args = parser.parse_args()
started = time.perf_counter()
features, adjacency, stage = citation_instance(args.graph, "cross_community_edges", 11)
rng = np.random.default_rng(np.random.SeedSequence([11, GRAPH_IDS[args.graph], stage]))
local, _, _ = extract(features, adjacency, Config(), rng)
h1 = residual_statistics(features, adjacency)[:, :1]
structural, _ = structural_statistics(features, adjacency)
statistics = np.concatenate([local, h1, structural], axis=1)
ranks, mask, counts, _ = exact_marginal(statistics, adjacency)
tails = ("upper", "upper", "upper", "lower", "two-sided",
         "upper", "lower", "upper", "upper")
score = score_bank(statistics, ranks, mask, counts,
                   tuple(range(6)), tuple(range(6, 9)), tails)
print(json.dumps({
    "method": "CoRe-GAD-MSDA", "graph": args.graph,
    "mechanism": "cross_community_edges", "seed": 11,
    "seconds": time.perf_counter() - started,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "trainable_parameters": 0, "training_required": False, "epochs": 0,
    "nodes": len(score), "finite_scores": bool(np.isfinite(score).all()),
}))
