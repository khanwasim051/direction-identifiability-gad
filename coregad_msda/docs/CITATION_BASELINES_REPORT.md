# Controlled four-mechanism comparison

All numeric entries below are measured results averaged over four mechanisms and five perturbation/training seeds (20 runs per graph).

| Method | Cora AUROC | Cora AP | CiteSeer AUROC | CiteSeer AP | PubMed AUROC | PubMed AP | Overall AUROC | Overall AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NeighborDiv | 0.5649 | 0.0717 | 0.6906 | 0.1325 | 0.7166 | 0.1525 | 0.6574 | 0.1189 |
| FreeGAD | 0.6640 | 0.1496 | 0.6970 | 0.1073 | 0.6970 | 0.1670 | 0.6860 | 0.1413 |
| DOMINANT | 0.6986 | 0.1457 | 0.7558 | 0.1617 | 0.7644 | 0.1299 | 0.7396 | 0.1458 |
| CoRe-GAD | 0.5814 | 0.0656 | 0.6887 | 0.0908 | 0.7108 | 0.1044 | 0.6603 | 0.0869 |

## Baselines not entered as numeric rows

- **ReFi-GAD:** its official test routine samples labeled normal and anomalous support nodes from the target graph (50 of each in its released configuration). It is therefore a few-shot comparator, not a label-free zero-shot comparator. A value from that routine must not be placed in the same table without an explicit few-shot marker and protocol column.
- **AnomalyGFM:** its official zero-shot code trains a model on a labeled source graph before target inference. The repository provides neither a citation-ready pretrained zero-shot checkpoint nor a uniquely determined source graph for this experiment. A source-domain protocol must be declared before running it; otherwise the result is under-specified.

## Interpretation constraint

DOMINANT is the strongest completed method in the aggregate table. These experiments therefore do not support a state-of-the-art performance claim for CoRe-GAD. They can support claims about label-free exact calibration, leakage control, and behavior across distinct structural mechanisms, provided those claims are stated separately from comparative superiority.

## Reproduction notes

- NeighborDiv: equation-faithful independent reproduction, rank 8, exhaustive unordered neighbor pairs, no target tuning.
- FreeGAD: equation-faithful reproduction of official commit `c1b007d69174e22fcf9d7ff7fea0dcb88ebdf2a0`; official Cora preset transferred unchanged to all citation graphs. A direct Torch cross-check gave maximum embedding error `2.98e-8` and maximum score error `2.55e-5`.
- DOMINANT: official PyGOD layers from commit `c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8`, fixed 20 epochs, with an algebraically exact sparse structure-error calculation. Dense-versus-sparse validation gave maximum score error `1.91e-6`.
- CoRe-GAD: frozen `structural_fixed_mean` results from the leakage-audited four-mechanism stress test.
