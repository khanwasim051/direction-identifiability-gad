# Baseline round 2: official code of GDN and BWGNN under the feasibility protocol
Written 2026-10-16, before any result under this protocol was computed. Same graphs (16), k (1,3,5,10 with
k <= floor(#anomalies/2)), seeds (11,23,37,53,71), anomaly shots (SeedSequence([seed, 11000 + k])) and evaluation
(AUROC/AP on all nodes except the shots) as FEASIBILITY_PROTOCOL.md. T2 graphs not used.

## Code
GDN: official repository kaize0409/Meta-GDN_AnomalyDetection (commit 426516b): models.SGC, meta.dev_loss,
     utils.sgc_precompute / normalize_adjacency (degree 2), raw features as in the official loader; training loop as
     in run.py: Adam lr 0.01, weight decay 0, 100 epochs, one balanced batch per epoch of the k labelled anomalies
     and k unlabelled nodes (test_task_generator_backup with bs = 2k).
BWGNN: official repository squareRoot3/Rethinking-Anomaly-Detection (commit de0631f): BWGNN.py unchanged
     (homo, order 2, hidden 64), raw features, Adam lr 0.01, 100 epochs, weighted cross-entropy as in main.py;
     training labels: the k shots = 1, all other nodes = 0 (unlabelled treated as normal).
Torch / numpy / random seeds = run seed.

## Two variants per baseline
(a) last epoch: no model selection (the few-shot setting has no labelled validation set). PRIMARY.
(b) labelled-validation selection, as in the official scripts (GDN: best validation AUROC; BWGNN: best validation
    macro-F1 with the official threshold search): validation = a stratified random 20% of the non-shot nodes with
    their labels (rng SeedSequence([seed, 12000 + k])). Uses many more labels than any other method; reported
    separately as an optimistic reference, evaluated on the same nodes as everything else.

## Use of results
Replace B1/B2 of the feasibility check by variant (a) and re-apply the same go/no-go criteria. Variant (b) is
reported, not used for the decision. Meta-GDN is not run: it needs labelled auxiliary graphs that share the
target's feature space, which this protocol does not provide (stated as a limitation).
