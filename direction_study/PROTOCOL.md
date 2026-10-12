# Pre-registered protocol: label-free per-graph direction selection for CoRe-GAD-MSDA
Written 2026-10-11 before any candidate rule was implemented or evaluated.

## What changes
Only the tail direction tau_k of each of the nine statistics. Statistics, exact neighborhood-excluded
calibration, reliability weights, branch-maximum fusion, and q, L are unchanged from the frozen
`directional_h1` detector. The direction of statistic k on graph G is chosen from the unlabeled
values {t_ik : t_ik finite} of that statistic on G only. No labels, no target tuning.

## Disclosure
The motivation (directions reverse on some standard graphs) was observed with labels on all
standard-suite graphs (dev and T1). Therefore T1 is "seen but not used for selection"; T2 is fully
untouched (never loaded with labels before the freeze) and is the primary held-out evidence.

## Splits
- DEV (labels used for rule selection):
  - Synthetic development runs of the original protocol: Cora and CiteSeer, seeds 11/23/37, four mechanisms (2 units).
  - Standard graphs: Cora (ARC injected), ACM, BlogCatalog, Amazon, Amazon-all, Tolokers (6 units).
- T1 (labels seen earlier, not used for selection): CiteSeer, PubMed, Flickr, Facebook, Reddit, Disney,
  Weibo (UNPrompt file), YelpChi, YelpChi-all, Questions; plus original synthetic held-out runs
  (Cora/CiteSeer seeds 53/71, PubMed seeds 11-71, four mechanisms).
- T2 (untouched): Amazon-Computers "cs" and Amazon-Photo "photo" (ARC injected files), Books and Enron (PyGOD organic).

Related graphs share a split (Amazon with Amazon-all; YelpChi with YelpChi-all).

## Candidate rules (complete list)
s_k = skewness of the finite values of statistic k on the graph.
  - "moment": bias-corrected sample skewness (scipy.stats.skew, bias=False).
  - "kelly":  (q90 + q10 - 2*q50) / (q90 - q10); 0 if q90 == q10.
Degenerate statistics (fewer than 3 finite values, or constant) keep the frozen direction.
- F0 frozen directions (reference).
- F1 all two-sided (reference).
- A(skew, tau): upper if s_k > tau, lower if s_k < -tau, two-sided otherwise.
  moment: tau in {0, 0.5, 1.0}; kelly: tau in {0, 0.1, 0.25}.
- B(skew, tau): frozen direction, flipped (upper<->lower) when sign(s_k) disagrees with it and |s_k| > tau;
  the two-sided statistic (mean log-neighborhood degree) becomes upper/lower by sign(s_k) when |s_k| > tau.
  Same tau grids as A.
- C consensus: start from frozen directions; up to 10 sweeps; each statistic takes the orientation in
  {upper, lower, two-sided} whose evidence has the largest Spearman correlation with the mean evidence of
  the other valid coordinates (over nodes where it is valid); stop when unchanged.
- D consensus initialized from A(kelly, 0.1) instead of the frozen directions.

## Selection criterion (on DEV only)
Unit utility U = 0.5 * [(AUROC - 0.5)/0.5 + (AP - pi)/(1 - pi)], averaged over seeds/runs within a unit;
score = mean over the 8 DEV units. Highest score wins; ties broken by mean DEV AUROC, then by simplicity
(A < B < C < D, smaller tau first). The winner is frozen in freeze.json before any T1/T2 evaluation.

## Reporting
The selected rule is "the method". All candidates are also reported on T1/T2 for transparency.
Seeds: 11, 23, 37, 53, 71 (standard graphs). Metrics: AUROC, AP.
