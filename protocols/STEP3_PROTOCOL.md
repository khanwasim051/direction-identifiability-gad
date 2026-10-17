# Step 3 (final development round): two targeted variants
Written 2026-10-17, before any result under this protocol was computed. Final round of method development on the
16 non-T2 graphs; after it the adopted method is frozen and tested once on the sealed T2 graphs.

## Results already seen (STEP2_PROTOCOL.md)
V2 (pooled relational + attribute statistics with multiplicity shrinkage) improved organic graphs (+0.045 mean
AUROC) and k >= 3, but lost 0.024 on injected graphs, mainly at k = 1 and on Flickr/BlogCatalog, so it was not
adopted. The two variants below address exactly that failure; nothing else changes.

## Setting, statistics and calibration
Identical to STEP2_PROTOCOL.md (same graphs, k, seeds, shots, statistics, calibration, shrinkage tau_k = 1/sqrt(12k)).
Notation: S_V0 = M2_DIR-W score; S_V2 = V2 score; S_ATT = V2's pooled score computed over the 2p attribute statistics
only (weights max(0, s_j - tau_k)); if every attribute weight is 0, S_ATT is undefined.

## Variants (exactly two)
W1  k-gated:   score = S_V2 if k >= 3, else S_V0.
W2  max-fusion: score = max(S_V0, S_ATT) per node; score = S_V0 where S_ATT is undefined.

## Selection rule (decided now)
Selected = the higher mean AUROC over all (graph, k) cells (cell = mean over 5 seeds) of W1 and W2. Adopted only if
its overall mean AUROC >= V0's AND its injected-graph mean AUROC >= V0's - 0.02. Otherwise V0 (M2_DIR-W) is the
final method. Whatever is adopted is frozen (code hashed) before any T2 label is read.
