# FitNets feature-hint report

- Paper: Romero et al., FitNets: Hints for Thin Deep Nets (ICLR 2015)
- Temperature: **3**
- Lambda curriculum: **4 -> 1**
- Float accepted: **True**
- Physical profiling complete: **False**

## Quality

| model | samples | threshold | accuracy | precision | recall | f1 | roc_auc | pr_auc | specificity | confusion_matrix | positive_prevalence | predicted_positive_rate | ece_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_control | 2000 | 0.395 | 0.57 | 0.538413 | 0.864424 | 0.663537 | 0.637925 | 0.61469 | 0.286555 | [[292, 727], [133, 848]] | 0.4905 | 0.7875 | 0.016454 |
| kd_control | 2000 | 0.365 | 0.545 | 0.520091 | 0.936799 | 0.66885 | 0.662779 | 0.642628 | 0.167812 | [[171, 848], [62, 919]] | 0.4905 | 0.8835 | 0.030474 |
| fitnets_hint_kd | 2000 | 0.18 | 0.6985 | 0.649762 | 0.835882 | 0.731164 | 0.80307 | 0.798469 | 0.566241 | [[577, 442], [161, 820]] | 0.4905 | 0.631 | 0.156829 |

## Representation diagnostics

| model | linear_cka | hint_mse_with_stage1_regressor |
| --- | --- | --- |
| shared_initial | 0.326586 | 4.203131 |
| hint_checkpoint | 0.956203 | 0.610759 |
| kd_control_final | 0.61719 | 1.586601 |
| fitnets_final | 0.917721 | 0.842921 |

## Acceptance gates

| pr_auc_gain | f1_retained | specificity_retained | feature_alignment_gain | graph_identity | accepted_float |
| --- | --- | --- | --- | --- | --- |
| True | True | True | True | True | True |
