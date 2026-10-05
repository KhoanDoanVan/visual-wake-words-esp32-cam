# Similarity-Preserving KD report

- Calibrated gamma: **18.6028**
- Internal causal acceptance: **False**
- Project champion acceptance: **False**
- Eligible for physical device: **False**
- Physical profiling complete: **False**

## Frozen-test quality

| model                 |   samples |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   specificity | confusion_matrix         |   positive_prevalence |   predicted_positive_rate |    ece_10 |
|:----------------------|----------:|------------:|-----------:|------------:|---------:|---------:|----------:|---------:|--------------:|:-------------------------|----------------------:|--------------------------:|----------:|
| sp_control            |      2000 |       0.305 |     0.7645 |     0.72973 | 0.825688 | 0.774749 |  0.861521 | 0.862016 |      0.705594 | [[719, 300], [171, 810]] |                0.4905 |                    0.555  | 0.0533891 |
| similarity_preserving |      2000 |       0.34  |     0.759  |     0.71162 | 0.85525  | 0.776852 |  0.863289 | 0.863601 |      0.66634  | [[679, 340], [142, 839]] |                0.4905 |                    0.5895 | 0.0162833 |

## Relation diagnostics

| model                 |   paper_relation_mse |   off_diagonal_relation_mse |   mean_same_class_similarity |   mean_cross_class_similarity |   class_relation_margin |
|:----------------------|---------------------:|----------------------------:|-----------------------------:|------------------------------:|------------------------:|
| teacher               |           0          |                  0          |                     0.154731 |                      0.119477 |               0.0352542 |
| sp_control            |           0.00322327 |                  0.00203376 |                     0.169289 |                      0.149879 |               0.0194095 |
| similarity_preserving |           0.00122727 |                  0.00115637 |                     0.154582 |                      0.126145 |               0.028436  |

## Project comparison

| model                 |   accuracy |   precision |   recall |   specificity |       f1 |   pr_auc |
|:----------------------|-----------:|------------:|---------:|--------------:|---------:|---------:|
| sp_control            |     0.7645 |    0.72973  | 0.825688 |      0.705594 | 0.774749 | 0.862016 |
| similarity_preserving |     0.759  |    0.71162  | 0.85525  |      0.66634  | 0.776852 | 0.863601 |
| notebook03_champion   |     0.7815 |    0.777551 | 0.776758 |      0.786065 | 0.777155 | 0.865103 |

## Gates

|                               |   passed |
|:------------------------------|---------:|
| internal_pr_auc_gain          |        1 |
| internal_f1_retained          |        1 |
| internal_specificity_retained |        0 |
| relation_alignment_gain       |        1 |
| graph_identity                |        1 |
| champion_pr_auc_matched       |        0 |
| champion_f1_matched           |        0 |
| accepted_internal             |        0 |
| accepted_project              |        0 |
| int8_parity                   |        0 |
| eligible_for_device           |        0 |
