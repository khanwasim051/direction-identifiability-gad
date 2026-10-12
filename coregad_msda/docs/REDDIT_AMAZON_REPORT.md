# Reddit/Amazon baseline and held-out evidence report

Weibo was excluded before these runs. All numbers below are measured results; no values were estimated or adjusted.

## Real-world comparison

| Method | Reddit AUROC | Reddit AP | Amazon AUROC | Amazon AP |
|---|---:|---:|---:|---:|
| NeighborDiv | 0.5108 | 0.0333 | 0.5472 | 0.1090 |
| FreeGAD | 0.5081 | 0.0358 | 0.3701 | 0.0692 |
| DOMINANT | 0.5603 | 0.0373 | 0.3892 | 0.0711 |
| AnomalyGFM (Facebook source) | 0.4277 | 0.0265 | 0.2002 | 0.0558 |
| CoRe-GAD-MSDA (frozen) | 0.5263 | 0.0345 | 0.7376 | 0.1879 |

NeighborDiv and FreeGAD are deterministic, so each has one run. DOMINANT, AnomalyGFM, and CoRe-GAD use five seeds. Amazon is scored on the conventional 8,639 labeled-node mask; Reddit uses all 10,984 nodes.

CoRe-GAD is strongest on Amazon (AUROC 0.7376, AP 0.1879), but it is not strongest on Reddit: DOMINANT reaches AUROC 0.5603 and AP 0.0373, compared with CoRe-GAD's 0.5263 and 0.0345. Therefore the defensible claim is target-dependent robustness, not universal superiority.

AnomalyGFM was trained for 301 epochs on the official feature-aligned Facebook source graph and transferred unchanged. Its poor transfer here must be reported as a fixed-source result, not as a general statement about the method. ReFi-GAD has no numeric row because its released target inference samples 50 labeled normal and 50 labeled anomalous nodes; that is few-shot support, not the label-free zero-shot setting evaluated here.

## Held-out evidence ablation

This was executed after the evidence bank had been frozen. It is explanatory only and was not used to change the selected method.

| Evidence bank | Held-out citation AUROC | Held-out citation AP | Reddit AUROC | Amazon AUROC |
|---|---:|---:|---:|---:|
| Original, two-sided | 0.6757 | 0.1153 | 0.4581 | 0.6421 |
| Direction-aware original | 0.7713 | 0.2180 | 0.5101 | 0.7161 |
| Direction-aware + h1 (frozen) | 0.7950 | 0.2623 | 0.5263 | 0.7376 |
| Direction-aware + h1+h2+h3 | 0.7833 | 0.2572 | 0.5356 | 0.7684 |

Direction awareness provides the largest consistent gain over the original two-sided evidence bank. Adding the one-hop residual then improves held-out citation performance further. Adding two- and three-hop residuals improves both organic targets but reduces the 36-run held-out citation mean from AUROC 0.7950 to 0.7833. Because this contrast was observed after freezing, the manuscript must retain `directional_h1` as the primary method and describe the deeper-scale result only as a post-freeze diagnostic.

## Validation

- Baseline metric rows: 14.
- AnomalyGFM metric rows: 10.
- Evidence-ablation rows: 322 = 46 graph instances × 7 evidence banks.
- Weibo metric rows: 0.
- Sparse-versus-dense AnomalyGFM forward maximum absolute error: 1.621e-05.
- All metrics are finite and within [0, 1]: True.
