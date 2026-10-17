# Feasibility check: few-shot direction calibration from labelled anomalies
Written 2026-10-16, before any result under this protocol was computed.

## Question
With k labelled anomalies and no labelled normal nodes, can CoRe-GAD-MSDA with calibrated statistic directions
match or beat few-shot GAD methods that are trained with the same k labelled anomalies?

## Prior exposure (disclosed)
Earlier results already seen: an AUROC-max direction rule using 10 anomalies + 10 normal nodes
(results_fewshot_cmp.csv), and the full-label direction diagnostic. The methods below were designed after seeing
those results. The four T2 graphs (Coauthor-CS, Amazon-Photo, Books, Enron) are NOT used here; their labels stay
sealed and are reserved for the confirmatory test of the final method.

## Supervision
k labelled anomalies, drawn uniformly without replacement from the anomalous nodes of the graph with
numpy default_rng(SeedSequence([seed, 11000 + k])). No labelled normal nodes. All other nodes are unlabelled.
k in {1, 3, 5, 10}; a (graph, k) cell is run only if k <= floor(#anomalies / 2).
Seeds 11, 23, 37, 53, 71 (seed also selects the cached CoRe-GAD-MSDA statistics for that seed).

## Graphs (16; all previously used non-T2 benchmark graphs, Weibo UNPrompt duplicate excluded)
Injected: Cora, CiteSeer, PubMed, ACM, BlogCatalog, Flickr, Facebook.
Organic: Reddit, Amazon, Disney, Weibo-ARC, Tolokers, Amazon-all, YelpChi-all, Questions, YelpChi.

## Methods (all fixed; no tuning on these graphs)
Notation: r_ij = cached graph-wide signed midrank in [-1, 1] of statistic j at node i; m_ij validity mask;
A = labelled anomalies valid for j; tails as in CoRe-GAD-MSDA (upper, lower, two-sided).
Estimated separation of tail t from the anomaly shots, against the unlabelled graph as reference:
  AUC_up = mean_{a in A} (r_aj + 1)/2,  AUC_low = 1 - AUC_up,  AUC_two = mean_{a in A} |r_aj|.
(r is the centred empirical CDF, so these are the Mann-Whitney AUCs of the shots against all nodes.)

M0 Frozen      : frozen directions (zero-shot reference).
M1 DIR         : direction_j = argmax_t AUC_t (ties -> frozen direction; A empty -> frozen). Detector otherwise unchanged.
M2 DIR-W (primary): directions as M1; each statistic's reliability weight is additionally multiplied by
                 s_j = max(0, AUC_best,j - 0.5). If s_j = 0 for every statistic of a branch, that branch scores 0.
B1 GDN         : Graph Deviation Network (Ding et al., WSDM 2021), re-implemented: 2-layer GCN (hidden 64) + linear
                 scorer; deviation loss with reference N(0,1) (5000 draws), margin a = 5; labelled anomalies vs
                 unlabelled nodes treated as normal; Adam lr 5e-3, weight decay 5e-4, 100 epochs, each epoch: all
                 anomaly shots + 512 random unlabelled nodes per step, 10 steps. Features row-normalised.
B2 BWGNN       : Beta Wavelet GNN (Tang et al., ICML 2022), re-implemented: order C = 2 (3 Beta-wavelet filters),
                 hidden 64; trained with cross-entropy on anomaly shots (label 1) vs unlabelled nodes (label 0),
                 positive class weight = #unlabelled / k; Adam lr 0.01, 100 epochs.
Baselines have no validation set (none exists with k anomalies), so they use fixed epochs. Torch seed = seed.

## Evaluation
AUROC and AP on all nodes except the shots. Mean over the 5 seeds per (graph, k).

## Go / no-go criteria (decided now)
GO for Route 2 if all of:
 (1) M2 mean AUROC over graphs >= mean AUROC of the better of B1/B2 - 0.01, at k = 5 and at k = 10;
 (2) M2 beats the better baseline on at least half of the graphs available at k = 10;
 (3) M2 mean AUROC > M0 mean AUROC at k = 5 and at k = 10.
Otherwise NO-GO (fall back to submitting the empirical paper).
Secondary (reported, not used for the decision): M1 vs M2; per-k curves; direction error of M1 against the
full-label best direction, compared with the Hoeffding bound exp(-2 k delta^2), delta = |AUC_full - 0.5|.
