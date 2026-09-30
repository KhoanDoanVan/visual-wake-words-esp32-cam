# Attention-transfer report

- Calibrated beta: **4.55377**
- Float accepted: **False**
- Physical profiling complete: **False**

## Quality

| model              |   samples |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   specificity | confusion_matrix         |   positive_prevalence |   predicted_positive_rate |    ece_10 |
|:-------------------|----------:|------------:|-----------:|------------:|---------:|---------:|----------:|---------:|--------------:|:-------------------------|----------------------:|--------------------------:|----------:|
| attention_control  |      2000 |       0.405 |      0.773 |    0.750714 | 0.804281 | 0.776575 |  0.860413 | 0.863304 |      0.742885 | [[757, 262], [192, 789]] |                0.4905 |                    0.5255 | 0.0165672 |
| attention_transfer |      2000 |       0.395 |      0.766 |    0.734247 | 0.819572 | 0.774566 |  0.861291 | 0.861767 |      0.714426 | [[728, 291], [177, 804]] |                0.4905 |                    0.5475 | 0.0158418 |

## Gates

|                      |   0 |
|:---------------------|----:|
| pr_auc_gain          |   0 |
| f1_retained          |   1 |
| specificity_retained |   0 |
| graph_identity       |   1 |
| accepted_float       |   0 |
