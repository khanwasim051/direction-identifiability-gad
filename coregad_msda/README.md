# CoRe-GAD-MSDA reproducibility package

This package contains the code, frozen configurations, run-level outputs, validation records, and manuscript-facing tables for **CoRe-GAD-MSDA**, a training-free zero-shot graph anomaly detector based on exact neighborhood-excluded calibration, direction-aware evidence, and reliability-weighted contextual/structural fusion.

All numeric values in `results/` are measured experiment outputs. No estimated or manually adjusted benchmark values are included.

## Final evaluation protocol

- Controlled citation graphs: Cora, CiteSeer, and PubMed.
- Controlled mechanisms: cross-community edges, degree-preserving rewiring, ego-edge deletion, and hub-star injection.
- Additional graphs: Reddit and Amazon with organic benchmark labels.
- Post-freeze external evidence: Facebook using the current TAM-distributed artifact; the original ConAD artifact is retained as a version-sensitivity analysis.
- Frozen method: `directional_h1` evidence with exact query-and-neighborhood-excluded signed midranks and reliability-weighted contextual/structural branch maximum.
- Target labels are used only after score construction for AUROC and average precision.
- Weibo is not part of the final manuscript protocol or manuscript-facing tables.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/` | Frozen evidence, aggregation, and final inclusion protocol |
| `results/manuscript/` | Tables corresponding directly to the Results section |
| `results/raw/` | Run-level metrics, protocols, completion files, and validation records |
| `results/node_scores/` | Per-node scores exported by `scripts/export_final_scores.py` |
| `condcal_cora/` | Citation-data loader and local contextual-statistic implementation |
| `condcal_citation_exact_global/` | Controlled perturbations, structural statistics, exact calibration, and injection records |
| `core_reliability_fusion/` | Reliability-weighted aggregation development |
| `core_multiscale_directional/` | Final CoRe-GAD-MSDA development, freezing, evaluation, and ablation |
| `baseline_comparison/` | NeighborDiv, FreeGAD, and DOMINANT citation runners |
| `realworld_benchmarks/` | Reddit/Amazon baseline runner |
| `anomalygfm_realworld/` | Sparse source-trained AnomalyGFM transfer reproduction |
| `facebook_evaluation/` | Frozen Facebook evaluation and sensitivity analysis |
| `docs/` | Evidence reports, protocol explanations, and publication guidance |
| `scripts/` | Score export and package-validation utilities |

## Environment

The recorded experiments used Python 3.12.14. Install the pinned dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The training-free CoRe-GAD-MSDA code requires NumPy, SciPy, and scikit-learn. PyTorch, PyTorch Geometric, and PyGOD are needed only for DOMINANT and AnomalyGFM.

## Prepare datasets

```bash
python condcal_cora/download_data.py
python condcal_amazon/prepare_data.py
```

Facebook and AnomalyGFM-aligned matrices require manual placement because they originate from third-party repositories. Exact filenames and SHA-256 checksums are listed in `DATASETS.md`. Raw datasets are intentionally excluded from this public-ready package.

## Reproduce the frozen method

Run development and freeze the evidence bank:

```bash
python core_multiscale_directional/run_experiment.py development \
  --output reproduced_results/core_msda
```

Verify that the selected bank matches `configs/frozen_evidence_bank.json`, then evaluate the frozen bank:

```bash
python core_multiscale_directional/run_experiment.py evaluation \
  --output reproduced_results/core_msda
```

The release version evaluates Cora, CiteSeer, PubMed, Reddit, and Amazon; it does not evaluate Weibo.

Run the post-freeze evidence ablation:

```bash
python core_multiscale_directional/run_heldout_evidence_ablation.py
```

## Reproduce baselines

```bash
python baseline_comparison/run_neighbor_div.py
python baseline_comparison/run_freegad.py
python baseline_comparison/run_dominant_exact.py
python realworld_benchmarks/run_available_baselines.py
```

For the source-trained AnomalyGFM comparison, place `Facebook_svd.mat` and `reddit_svd.mat` in `external_data/`, then run:

```bash
COREGAD_EXTERNAL_DATA=external_data \
python anomalygfm_realworld/run_sparse_zero_shot.py
```

For Facebook, place `Facebook.mat` and `fb_conad_full.mat` in `facebook_evaluation/data/`, then run:

```bash
python facebook_evaluation/run_facebook.py
```

## Export per-node scores

After preparing the datasets, generate compressed per-node score files for every final dataset and controlled mechanism:

```bash
python scripts/export_final_scores.py --scope all
```

For the evidence bundle supplied with this release, these score files have already been generated. They allow a reviewer to recompute metrics independently from node labels and anomaly scores.

Recompute AUROC and AP directly from those exported files:

```bash
python scripts/recompute_metrics.py
```

## Validate the package

```bash
python scripts/validate_release.py
```

This checks frozen configuration identifiers, headline manuscript values, file completeness, metric ranges, absence of Weibo from final tables, and SHA-256 integrity.

## Baseline-protocol qualifications

- NeighborDiv is an equation-faithful implementation using rank 8 and exhaustive unordered neighbor pairs.
- FreeGAD follows official commit `c1b007d69174e22fcf9d7ff7fea0dcb88ebdf2a0`, with the official Cora preset transferred unchanged.
- DOMINANT uses official PyGOD layers from commit `c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8` and an algebraically exact sparse structure-error calculation.
- AnomalyGFM is an independent sparse refactor trained on the official feature-aligned Facebook source and transferred to Reddit/Amazon.
- ReFi-GAD is documented but has no numeric row because its released inference routine samples 50 labeled normal and 50 labeled anomalous target nodes; that protocol is few-shot rather than label-free zero-shot.

## Supported interpretation

The evidence supports superior performance among the locally executed citation baselines, strong Amazon and Facebook performance, and target-dependent robustness on additional graphs. It does not support universal superiority: DOMINANT is stronger on Reddit, and the published TAM Facebook AUROC exceeds the local CoRe-GAD-MSDA result. The four controlled mechanisms were used during both development and evaluation, so the study demonstrates held-out seed/dataset transfer, not unseen-mechanism generalization.

Before making the repository public, replace the placeholders in `CITATION.cff.template`, choose a software license, and archive the tagged release using Zenodo or an equivalent repository.
