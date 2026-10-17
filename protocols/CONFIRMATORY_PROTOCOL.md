# Confirmatory test of the frozen few-shot method on the held-out graphs
Written 2026-10-17, after the method was frozen (METHOD_FREEZE.json) and before any held-out label was read for
this method. The held-out graphs were evaluated once before, in the earlier label-free direction-rule study
(PROTOCOL.md, T2 split); they were not used in developing the few-shot method.

## Graphs
Coauthor-CS (cs.mat), Amazon-Photo (photo.mat), Books and Enron (PyGOD), as in the T2 split.

## Setting (identical to FEASIBILITY_PROTOCOL.md / STEP3_PROTOCOL.md)
k in {1,3,5,10} with k <= floor(#anomalies/2); seeds 11,23,37,53,71; anomaly shots SeedSequence([seed, 11000+k]);
no labelled normal nodes; AUROC and AP on all nodes except the shots; cell = (graph, k), mean over seeds.

## Methods
FINAL   frozen W1_kgated (step3.final_scores, hashes in METHOD_FREEZE.json); V0 (M2_DIR-W) reported alongside.
GDN     official code, variants (a) last epoch and (b) labelled-validation selection, as BASELINES2_PROTOCOL.md.
BWGNN   official code, variants (a) and (b), as BASELINES2_PROTOCOL.md (run on the author's laptop with the same
        package, data for these four graphs only).

## Confirmatory claims (decided now)
C1  Mean AUROC of FINAL over all cells >= mean over cells of the better baseline per cell (variant a) - 0.01.
C2  FINAL beats the better baseline (variant a) in at least half of the cells.
C3  Mean AUROC of FINAL >= mean AUROC of V0.
All three are reported as passed or failed; the method is not changed after this test, whatever the outcome.
