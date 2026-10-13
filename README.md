# Investigating the Role of Direction Identifiability of Relational Statistics in the Performance of Training-Free Graph Anomaly Detection

Code, pre-registered protocols, and results for the paper

> W. Khan, S. A. Waheed, S. R. Wakchaure, A. D. Patil, D. Varshney, V. Khan.
> *Investigating the Role of Direction Identifiability of Relational Statistics in the Performance of Training-Free Graph Anomaly Detection.* Submitted to Data Mining and Knowledge Discovery.

Training-free graph anomaly detectors score nodes with relational statistics, and each statistic must be oriented: someone decides whether high or low values are anomalous. Using the nine-statistic detector **CoRe-GAD-MSDA** as an instrument, the paper shows that this direction decides performance, that it cannot be identified from unlabelled data, that labelled normal nodes do not recover it, and that a few labelled anomalies do.

## Repository layout

| Folder | Contents | Paper |
|---|---|---|
| `coregad_msda/` | Original CoRe-GAD-MSDA release: detector, synthetic perturbations, baselines, frozen configuration, run-level results | Sec. 3, RQ1 |
| `benchmarks/` | Standard-benchmark loaders, exact vectorised statistics, CoRe-GAD-MSDA and baseline runners, aggregation | RQ1, RQ2 |
| `direction_study/` | Pre-registered protocols (with SHA-256), label-free direction rules, evaluator, few-shot experiments | RQ3–RQ5 |
| `paper/` | Scripts that generate every number, table and figure; LaTeX source of the manuscript | all |
| `results/` | All results reported in the paper (metrics, per-node scores, logs, summaries) | all |
| `data/` | `download_data.py` only; datasets are downloaded, not redistributed | |
| `paths.py` | Repository paths (`GAD_DATA` and `GAD_WORK` environment variables override the data and work folders) | |

## 1. Installation

Python 3.11 or newer and `git` are required.

```bash
git clone https://github.com/khanwasim051/direction-identifiability-gad.git
cd direction-identifiability-gad
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CoRe-GAD-MSDA itself needs only NumPy, SciPy and scikit-learn; PyTorch, PyG and PyGOD are needed only for the baselines. Everything runs on a CPU.

## 2. Data

```bash
python data/download_data.py
```

This clones the public repositories of ARC, UNPrompt, AnomalyGFM, CARE-GNN and the PyGOD data collection into `data/`, extracts the archives, and downloads the Planetoid citation files used by the original release. The Facebook files for `coregad_msda/facebook_evaluation` must be placed by hand, as described in `coregad_msda/DATASETS.md`.

Two benchmarks exist in incompatible versions (Reddit: PyGOD `.pt` versus TAM/ARC `.mat` features; Weibo: ARC versus UNPrompt files). `benchmarks/bench_data.py` states which file every dataset name refers to.

## 3. Reproducing the paper

Run all commands from the repository root. Times are for one CPU core.

### RQ1: Reproduction and exactness

```bash
python benchmarks/repro.py          # reruns the released frozen evaluation; prints MAX_ABS_DIFF (expected 0.0), ~5 min
python benchmarks/verify_fast.py    # checks the vectorised statistics against the original loops
python coregad_msda/scripts/validate_release.py
```

### RQ2: Standard benchmarks

```bash
python benchmarks/core_bench.py Cora CiteSeer PubMed ACM BlogCatalog Flickr Facebook \
    Reddit Amazon Disney Weibo Tolokers Amazon-all YelpChi-all Questions YelpChi
python benchmarks/tf_baselines.py Cora CiteSeer PubMed ACM BlogCatalog Flickr Facebook \
    Reddit Amazon Disney Weibo Tolokers Amazon-all YelpChi-all Questions YelpChi
python benchmarks/prepare_generalist_data.py
python benchmarks/run_unprompt.py Amazon,Reddit,Weibo,YelpChi,Amazon-all,YelpChi-all,Disney,Tolokers,Questions 11,23,37,53,71
python benchmarks/run_anomalygfm.py Disney Amazon Reddit Weibo Tolokers Amazon-all
python benchmarks/aggregate.py      # writes results/benchmarks/all_results.json
```

CoRe-GAD-MSDA takes from about 1 s (Cora) to 3 min (Amazon-all) per seed. UNPrompt and AnomalyGFM are run with their official code, trained on Facebook; AnomalyGFM's zero-shot code needs a dense adjacency matrix and therefore enough RAM for graphs of about 12,000 nodes.

### RQ3–RQ4: Direction diagnostic and pre-registered label-free rules

```bash
python direction_study/cache.py synthetic Cora CiteSeer PubMed ACM BlogCatalog Flickr Facebook Reddit Amazon \
    Disney Weibo Weibo-ARC Tolokers Amazon-all YelpChi-all Questions YelpChi CS Photo Books Enron   # ~20 min
python direction_study/evaluate.py dev      # rule selection on development graphs
python direction_study/evaluate.py t1       # seen test graphs
python direction_study/evaluate.py t2       # untouched test graphs (refuses to run without results/direction_study/freeze.json)
```

`direction_study/PROTOCOL.md` lists the splits, the 16 candidate rules and the selection criterion; its SHA-256 is in `PROTOCOL.sha256` and in the paper (Appendix B). The freeze record `results/direction_study/freeze.json` stores the hashes of the rule and evaluator code at the time of freezing; the files with exactly those hashes are kept in `direction_study/frozen_originals/` (the working copies differ only in their path handling). Verify with:

```bash
sha256sum direction_study/PROTOCOL.md direction_study/frozen_originals/*.py
```

### RQ5: Few-shot direction selection

```bash
python direction_study/fewshot.py           # feasibility check (FEWSHOT_PROTOCOL.md)
python direction_study/fewshot_cmp.py       # comparison with ARC and AnomalyGFM (FEWSHOT_COMPARISON_PROTOCOL.md)
```

### Tables, figures and manuscript

```bash
python paper/paper_numbers.py   # every number in the paper -> results/paper/numbers.json (needs the cache from RQ3)
python paper/make_tables.py     # LaTeX tables -> paper/latex/gen/
python paper/make_figs.py       # figures -> paper/latex/fig/
python paper/build.py           # inlines the tables into paper/latex/main.tex
cd paper/latex && latexmk -pdf main.tex
```

`make_tables.py` and `make_figs.py` run directly on the shipped `results/paper/numbers.json`, so the tables and figures can be regenerated without rerunning any experiment.

## 4. Where each result comes from

| Paper item | File |
|---|---|
| Table 3 (organic suite), Table 4 (injected suite) | `results/benchmarks/all_results.json`, `results/paper/numbers.json` |
| Fig. 2 (direction diagnostic) | `numbers.json` → `diag` |
| Table 5, Fig. 3 (16 label-free rules) | `results/direction_study/summary_{dev,t1,t2}.json`, `results_{dev,t1,t2}.csv`, `freeze.json` |
| Tables 6–7 (few-shot comparisons) | `results/direction_study/results_fewshot_cmp.csv` |
| Fig. 4 (direction recovery) | `numbers.json` → `recovery` |
| Per-node scores of CoRe-GAD-MSDA | `results/benchmarks/core_out/scores_*.npz`, `coregad_msda/results/node_scores/` |

## 5. Notes

- Published baseline values (ARC, AnomalyGFM few-shot, TAM, DOMINANT, CoLA) are taken from the original papers and are marked as such in the paper.
- The statistics on very dense graphs use exact closed forms (`benchmarks/fast_stats.py`); they match the original per-node loops to within 6e-7.
- The held-out file `cs.mat` is Coauthor-CS; the RQ4 protocol calls it Amazon-Computers by mistake (see the paper, Appendix B).

## Citation

See `CITATION.cff`. A BibTeX entry will be added on publication.

## Licence

Code: MIT (see `LICENSE`). Datasets and third-party code are downloaded from their original sources and keep their own licences.
