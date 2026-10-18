# Few-Shot Direction Calibration of Relational and Attribute Statistics for Training-Free Graph Anomaly Detection

Code, protocols and results for the paper

> W. Khan, S. A. Waheed, S. R. Wakchaure, A. D. Patil, D. Varshney, V. Khan.
> *Few-Shot Direction Calibration of Relational and Attribute Statistics for Training-Free Graph Anomaly Detection.*
> Submitted to Knowledge and Information Systems.

Training-free graph anomaly detectors score nodes with relational statistics and must decide, for each statistic, whether high or low values are anomalous. The paper proves that this direction cannot be identified from unlabelled data, that labelled normal nodes carry little information about it when anomalies are rare, and that a few labelled anomalies identify it at an exponential rate. **FDC** (few-shot direction calibration) uses only *k* labelled anomalies to orient and weight relational and attribute statistics, with no training. It is compared with the official GDN and BWGNN under identical supervision on 16 development graphs and, after the method was frozen, on 4 held-out graphs.

## Repository layout

| Folder | Contents |
|---|---|
| `coregad_msda/` | Base detector CoRe-GAD-MSDA (original release); FDC uses its statistics and rank calibration |
| `gad/` | Benchmark loaders, exact vectorised statistics, and `statistics_cache.py` (statistics per graph and seed) |
| `fdc/` | FDC and all experiment scripts (development rounds, official GDN, confirmatory test, tables, runtime) |
| `bwgnn_package/` | Self-contained runner for the official BWGNN (needs DGL; was run on Windows) |
| `protocols/` | All protocols with SHA-256 hashes, the method freeze record, and the frozen method files |
| `results/` | Every per-run result used in the paper (`results/revision/`: analyses added in the revision) |
| `paper/` | Script that generates every table and figure, and the LaTeX source |
| `data/` | `download_data.py` only; datasets are downloaded, not redistributed |
| `paths.py` | Paths (`GAD_DATA` and `GAD_WORK` override the data and work folders) |

## 1. Installation

```bash
git clone https://github.com/khanwasim051/direction-identifiability-gad.git
cd direction-identifiability-gad
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data and official baseline code

```bash
python data/download_data.py              # benchmark files at the pinned repository commits; checks every SHA-256 (Appendix C)
python fdc/get_official_baselines.py      # official GDN and BWGNN code at the commits used in the paper
```

## 3. Reproducing the paper

Run from the repository root.

```bash
# statistics of the base detector for every graph and seed (held-out labels are stored separately)
python gad/statistics_cache.py Cora CiteSeer PubMed ACM BlogCatalog Flickr Facebook Reddit Amazon Disney \
    Weibo-ARC Tolokers Amazon-all YelpChi-all Questions YelpChi CS Photo Books Enron

# development round 1 (FEASIBILITY_PROTOCOL.md): calibration variants and the go/no-go check
python fdc/fsdc.py ours
python fdc/fsdc.py base Cora CiteSeer ...      # re-implemented baselines used only in round 1
python fdc/analyse.py

# official baselines on the development graphs (BASELINES2_PROTOCOL.md)
python fdc/run_gdn_official.py Cora CiteSeer PubMed ACM BlogCatalog Flickr Facebook Reddit Amazon Disney \
    Weibo-ARC Tolokers Amazon-all YelpChi-all Questions YelpChi
python bwgnn_package/make_bwgnn_data.py         # then run bwgnn_package/run_bwgnn.py dev (see below)

# development rounds 2 and 3 (STEP2_PROTOCOL.md, STEP3_PROTOCOL.md)
python fdc/step2.py
python fdc/step3.py

# confirmatory test on the held-out graphs (CONFIRMATORY_PROTOCOL.md)
python fdc/confirm.py                            # refuses to run if any frozen file has changed
python fdc/run_gdn_official.py --heldout Books Enron Photo CS
#   bwgnn_package/run_bwgnn.py heldout

# summary tables, bound check, runtime
python fdc/tables.py
python fdc/runtime.py Cora CiteSeer ...

# revision analyses (frozen method unchanged): 15 extra seeds, lightweight baselines, nAP and R-precision,
# null experiment with random shots, sensitivity to c / k-gate / p / (q, L), official GDN with 20 seeds,
# graph-level significance with hierarchical bootstrap
python fdc/rev_extra_seeds.py Cora CiteSeer ...      # statistics for seeds 101-115
python fdc/rev_eval.py Cora CiteSeer ...             # all 20 graphs
python fdc/rev_qL.py Amazon BlogCatalog CiteSeer Cora Tolokers Weibo-ARC YelpChi
python fdc/rev_gdn20.py Cora CiteSeer ...            # all 20 graphs
python fdc/rev_graph_level.py                      # graph-level tests; unrounded and paired per-run differences
python fdc/rev_runtime.py Disney Books ...          # runtime, same timing scope for FDC and GDN
python fdc/rev_missing_shots.py                     # how often |S_j| < k; threshold variant with tau_|S_j| (not frozen)
python paper/make_graphlevel_table.py && python paper/make_revision_tables.py
python paper/make_fig1.py && python paper/make_fig2.py && python paper/make_file_table.py

# tables, figure and manuscript
python paper/make_kais.py && python paper/build.py
cd paper/latex && latexmk -pdf main.tex
```

`paper/make_kais.py` reads only the files in `results/`, so the tables and figures can be regenerated without rerunning any experiment.

### Official BWGNN

The official BWGNN code uses DGL and runs on the CPU. DGL 2.2.1 provides ready-made packages for Windows only, so BWGNN was run on a Windows laptop:

```bash
cd bwgnn_package
conda create -n gad python=3.11 -c conda-forge --override-channels -y && conda activate gad
python -m pip install -r requirements.txt
python run_bwgnn.py check        # downloads the official BWGNN.py, verifies its SHA-256, runs a short test
python run_bwgnn.py dev          # 16 development graphs
python run_bwgnn.py heldout      # 4 held-out graphs
```

Copy the resulting `results/bwgnn_*.csv` files to `results/official_dev/bwgnn/` or `results/heldout/bwgnn/`.

## 4. Protocols and freezing

Every stage was governed by a protocol written and hashed before its results were computed (`protocols/`):

| Stage | Protocol |
|---|---|
| Round 1: relational calibration, go/no-go | `FEASIBILITY_PROTOCOL.md` |
| Official GDN and BWGNN | `BASELINES2_PROTOCOL.md` |
| Round 2: attribute statistics (best variant not adopted) | `STEP2_PROTOCOL.md` |
| Round 3: final variants (k-gated FDC adopted) | `STEP3_PROTOCOL.md` |
| Confirmatory test on held-out graphs | `CONFIRMATORY_PROTOCOL.md` |

`protocols/METHOD_FREEZE.json` records the SHA-256 of the method code and of all protocols at the moment the method was frozen, before any held-out label was read. The hashed method files are kept verbatim in `protocols/frozen/`. The working copies in `fdc/` differ from them only in file paths:

```bash
sha256sum protocols/*.md protocols/frozen/*.py
diff protocols/frozen/step3.py fdc/step3.py
```

## 5. Where each result comes from

| Paper item | File |
|---|---|
| Development results by *k*, per-graph table, Fig. 3 | `results/rounds23/step3.csv`, `results/official_dev/`, `results/tables/per_cell_mean_std.csv` |
| Confirmatory test | `results/heldout/`, `results/heldout/confirmatory_summary.json` |
| Graph-level significance (Table 7) | `results/revision/graph_level.csv`, `results/revision/graph_level_per_graph.csv` |
| Lightweight baselines, 20 seeds, nAP and R-precision (Tables 9, 10) | `results/revision/eval_*.csv`, `results/revision/gdn20_*.csv` |
| Null experiment (Table 11) | `results/revision/null_*.csv` |
| Sensitivity to c, k-gate, p (Table 12) and (q, L) (Table 13) | `results/revision/eval_*.csv`, `results/revision/qL_*.csv` |
| Ablation and development history | `results/round1/ours.csv`, `results/rounds23/step2.csv`, `results/rounds23/step3.csv` |
| Bound check (theory section) | `results/tables/bound_check.csv` |
| Runtime (Table 14) | `results/revision/runtime_rev.csv`, `results/revision/runtime/` |
| Direction diagnostic (Fig. 2, Table D1) | `results/tables/direction_diagnostic.csv` |
| Benchmark files (Table C1) | `data/download_data.py` (commits and SHA-256) |

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Datasets and third-party code are downloaded from their original sources and keep their own licences.
