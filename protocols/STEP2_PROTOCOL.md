# Step 2: extending few-shot direction calibration to attribute statistics
Written 2026-10-17, before any result under this protocol was computed.

## Motivation (from results already seen)
With k labelled anomalies, calibrated CoRe-GAD-MSDA (M2_DIR-W) beats official GDN and BWGNN on injected-anomaly
graphs but loses on several organic ones (Amazon, Amazon-all, Tolokers, Reddit, Questions, YelpChi-all, Facebook),
where supervised baselines exploit node attributes. Step 2 adds attribute statistics to the same calibration rule.
Design choices were made after seeing those per-graph results; the four T2 graphs (Coauthor-CS, Amazon-Photo,
Books, Enron) remain sealed for the confirmatory test of the frozen method.

## Setting (unchanged)
16 graphs, k in {1,3,5,10} with k <= floor(#anomalies/2), seeds 11,23,37,53,71, anomaly shots
SeedSequence([seed, 11000+k]), no labelled normal nodes, AUROC/AP on all nodes except the shots.

## Statistics
Relational (9): cached CoRe-GAD-MSDA statistics of that seed, with validity masks, graph-wide signed midranks
(R.marginal_ranks), as in FEASIBILITY_PROTOCOL.md.
Attribute (2p, p = min(16, d - 1)):
  Z = node features; if d > 64 or density < 0.3: Z = TruncatedSVD(p, random_state=0) of row-normalised features,
  else: Z = first p principal components (PCA, random_state=0) of column-standardised features.
  a_1..a_p   = the p columns of Z (own attributes);
  b_1..b_p   = mean of Z over neighbours (0 for isolated nodes) (neighbourhood attributes).
  All valid for every node; graph-wide signed midranks.

## Calibration rule (same as M2 for every statistic j)
AUC_up,j = mean over valid shots of (r_aj+1)/2; AUC_low,j = 1 - AUC_up,j; AUC_two,j = mean |r_aj|.
tail_j = argmax; s_j = max(0, AUC_best,j - 0.5). Evidence e_ij = oriented rank in [0,1] (0 if invalid).

## Variants (all fixed; exactly these four are run)
V0  M2_DIR-W unchanged (reference).
V1  pooled: score_i = sum_j w_j e_ij m_ij / sum_j w_j m_ij over all 9 + 2p statistics, w_j = s_j.
V2  V1 with multiplicity shrinkage: w_j = max(0, s_j - tau_k), tau_k = 1 / sqrt(12 k) (one null standard deviation
    of the mean of k uniform ranks). If all w_j = 0, fall back to V1 weights.
V3  V2 restricted to the top-m statistics by s_j, m = 8 (sparse selection).

## Selection rule (decided now)
Selected variant = highest mean AUROC over all (graph, k) cells (each cell = mean over the 5 seeds), among V1-V3.
It is adopted only if its mean AUROC is >= V0's mean AUROC AND its mean AUROC over the 7 injected graphs is not more
than 0.02 below V0's. Otherwise V0 stays the method. The adopted variant is then frozen (code hashed) for the T2 test.

## Reported (not used for selection)
Organic vs injected means; comparison with official GDN and BWGNN (BASELINES2_PROTOCOL.md, variants a and b); AP.
