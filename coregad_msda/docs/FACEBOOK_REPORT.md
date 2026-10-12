# Frozen CoRe-GAD Evaluation on Facebook

## Decision

The Facebook result is favorable and may be included as **supportive post-freeze
external evidence**, but it does not establish universal or state-of-the-art
superiority. The frozen CoRe-GAD-MSDA configuration outperformed the three
locally executed baselines on both public Facebook artifacts without
target-specific tuning. However, the published TAM result is higher in AUROC,
and the public dataset artifacts disagree with the commonly reported metadata.

## Primary result: current TAM-distributed artifact

The unmodified file contains 1,081 nodes, 55,104 symmetric adjacency entries
(27,552 undirected edges), 576 features, and 25 anomalies (2.31%).

| Method | Runs | AUROC | AP |
|---|---:|---:|---:|
| **CoRe-GAD-MSDA (frozen)** | 5 | **0.8778 ± 0.0004** | **0.1939 ± 0.0021** |
| NeighborDiv | 1 | 0.6117 | 0.0496 |
| FreeGAD | 1 | 0.4852 | 0.0278 |
| DOMINANT | 5 | 0.1134 ± 0.0033 | 0.0132 ± 0.0000 |

For the mean CoRe-GAD score across seeds, the stratified node-bootstrap 95%
intervals are [0.8078, 0.9393] for AUROC and [0.1276, 0.3402] for AP. The wide
AP interval reflects the presence of only 25 positive nodes.

## Version-sensitivity result: original ConAD artifact

The original ConAD file contains 1,086 nodes, the same 55,104 symmetric
adjacency entries, 576 features, and 27 anomalies (2.49%).

| Method | Runs | AUROC | AP |
|---|---:|---:|---:|
| **CoRe-GAD-MSDA (frozen)** | 5 | **0.8839 ± 0.0005** | **0.2265 ± 0.0019** |
| NeighborDiv | 1 | 0.6140 | 0.0499 |
| FreeGAD | 1 | 0.4504 | 0.0276 |
| DOMINANT | 5 | 0.1082 ± 0.0030 | 0.0141 ± 0.0000 |

The comparable performance across the two artifacts indicates that the main
CoRe-GAD conclusion is not caused by choosing one of these public versions.

## Leakage and reproducibility audit

- The evidence bank remained `directional_h1`, selected before this Facebook
  evaluation.
- Exact query-and-neighborhood-excluded calibration and the frozen
  reliability-weighted branch-maximum aggregation were retained unchanged.
- Target labels were not supplied to preprocessing, statistic construction,
  calibration, aggregation, or any baseline scoring function.
- No target-specific hyperparameter was selected.
- Five fixed seeds (11, 23, 37, 53, and 71) were used for stochastic methods.
- All 24 metric rows were finite and in the valid [0,1] interval.
- Every Facebook node received calibrated evidence; no zero-evidence node was
  observed.

## Manuscript use

Use the current TAM-distributed artifact as the primary reproducible result and
report its observed 25 anomalies (2.31%). State explicitly that this differs
from the 27 anomalies (2.49%) written in the TAM README. Place Facebook in the
injected-anomaly-on-a-real-graph subsection, consistent with the TAM
repository's experimental description.

The defensible claim is:

> On the Facebook graph, the frozen CoRe-GAD configuration achieved 0.8778
> AUROC and 0.1939 AP and substantially outperformed the locally executed
> NeighborDiv, FreeGAD, and DOMINANT configurations without target-specific
> tuning. A sensitivity analysis using the original ConAD artifact produced
> consistent results (0.8839 AUROC and 0.2265 AP).

Do not claim that CoRe-GAD is the best published method on Facebook. TAM reports
0.9144 AUROC and 0.2233 AUPRC, although its reported 27-label dataset does not
match the current 25-label repository artifact exactly. This should be treated
as a published-reference comparison rather than an identical-data rerun.
