# Pre-specified few-shot comparison with ARC and AnomalyGFM
Written 2026-10-11 before any result under this protocol was computed.

## Supervision (matches the baselines)
ARC (NeurIPS'24) and AnomalyGFM (KDD'25) use k labeled NORMAL target nodes only. Primary comparison: k labeled
normal nodes, no labeled anomalies. k = 10 (primary), plus 1 and 5 (AnomalyGFM table).

## Our method under this supervision (single rule, no selection)
Frozen CoRe-GAD-MSDA pipeline; only tail directions change. For statistic j with signed ranks r_ij and validity mask:
  N = valid labeled normal shots; U = valid unlabeled nodes.
  up  = fraction of U with r_ij > max_{N} r_ij;  low = fraction of U with r_ij < min_{N} r_ij.
  direction = upper if up > low; lower if low > up; frozen direction if equal or if N is empty.
Name: "normal-shot excess-tail rule". No thresholds, no tuning.

## Runs and evaluation
Seeds 11, 23, 37, 53, 71 (each seed = one draw of shots with
numpy default_rng(SeedSequence([seed, 7000 + k])) uniformly from normal nodes, and the method's cached
statistics for that seed). Metrics AUROC and AP on all nodes except the shots; mean over the 5 seeds.
Frozen zero-shot directions are evaluated on the same nodes.

## Datasets and comparisons
- ARC comparison (ARC's files, ARC Table 1, 10-shot): Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo (ARC file), Reddit, Amazon.
- AnomalyGFM comparison (files as in AnomalyGFM/UNPrompt, Table 3, k = 1/5/10): Reddit, Amazon, Disney, Amazon-all,
  YelpChi-all, Tolokers, Questions. (Elliptic, T-Finance not obtainable here.)
Baseline numbers are the published values (same benchmark files; protocol differences noted in the paper).

## Secondary (different supervision, reported separately)
k labeled anomalies + k labeled normals, AUROC-max rule as in FEWSHOT_PROTOCOL.md, k = 10, same datasets.
