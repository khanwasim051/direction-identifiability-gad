# Frozen method-development record

## Aggregation development

The aggregation rule was developed on Cora and CiteSeer using seeds 11, 23, and 37 under four controlled mechanisms. Candidate rules included a fixed mean, branch maximum, reliability-weighted mean, weighted Stouffer combination, weighted Cauchy combination, reliability-adaptive maximum, reliability-weighted branch maximum, and branch-wise Cauchy maximum. The selected rule, `reliability_branch_max`, was serialized before held-out evaluation. Reliability is the product of reference-support, evidence-dispersion, and local-versus-marginal calibration-stability factors.

## Evidence development

With exact neighborhood-excluded calibration and the aggregation rule fixed, the evidence study compared semantic directionality and propagation-residual scales. The one-hop direction-aware bank, `directional_h1`, achieved the highest development utility and was frozen before evaluating Cora/CiteSeer held-out seeds, PubMed, Reddit, Amazon, or Facebook.

| Evidence bank | Development AUROC | Development AP | Utility |
|---|---:|---:|---:|
| directional_h1 | 0.7753 | 0.2298 | 0.3699 |
| directional_h12 | 0.7611 | 0.2266 | 0.3541 |
| directional_h123 | 0.7557 | 0.2205 | 0.3455 |
| directional_h123_degree | 0.7526 | 0.2110 | 0.3374 |
| directional_original | 0.7443 | 0.1444 | 0.2941 |
| twosided_h123_degree | 0.6552 | 0.0869 | 0.1747 |
| original_twosided | 0.6379 | 0.0797 | 0.1535 |

The result supports semantic tail orientation and a one-hop propagation residual. Deeper propagation scales were not selected because they reduced citation development utility. The later held-out ablation is explanatory only and did not alter the frozen method.

## Final inclusion policy

The manuscript retains Cora, CiteSeer, PubMed, Reddit, Amazon, and the post-freeze Facebook evaluation. Weibo was excluded from the final protocol. Its earlier diagnostic outcome is not used in manuscript tables, aggregate claims, or public score exports.
