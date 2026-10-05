# Quantization-aware Knowledge Distillation report

- Full QKD desktop accepted: **False**
- Eligible for physical profiling: **False**
- Physical profiling complete: **False**

## Frozen-test quality

| model                     |   samples |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   specificity | confusion_matrix         |   positive_prevalence |   predicted_positive_rate |    ece_10 |
|:--------------------------|----------:|------------:|-----------:|------------:|---------:|---------:|----------:|---------:|--------------:|:-------------------------|----------------------:|--------------------------:|----------:|
| float_champion            |      2000 |       0.385 |     0.7815 |    0.777551 | 0.776758 | 0.777155 |  0.862603 | 0.865103 |     0.786065  | [[801, 218], [219, 762]] |                0.4905 |                    0.49   | 0.0394337 |
| ptq_int8                  |      2000 |       0.275 |     0.7615 |    0.745614 | 0.779817 | 0.762332 |  0.853138 | 0.856849 |     0.743867  | [[758, 261], [216, 765]] |                0.4905 |                    0.513  | 0.098125  |
| hard_qat_int8             |      2000 |       0.01  |     0.4905 |    0.4905   | 1        | 0.658168 |  0.547356 | 0.522209 |     0         | [[0, 1019], [0, 981]]    |                0.4905 |                    1      | 0.425457  |
| fixed_teacher_qat_kd_int8 |      2000 |       0.025 |     0.4975 |    0.493933 | 0.995923 | 0.660358 |  0.544842 | 0.525732 |     0.0176644 | [[18, 1001], [4, 977]]   |                0.4905 |                    0.989  | 0.399563  |
| qkd_full_int8             |      2000 |       0.05  |     0.496  |    0.493144 | 0.989806 | 0.658305 |  0.538105 | 0.522336 |     0.0206084 | [[21, 998], [10, 971]]   |                0.4905 |                    0.9845 | 0.37475   |

## Paired bootstrap: full QKD minus hard QAT

| metric       |        mean |      ci_low |    ci_high |   probability_positive |
|:-------------|------------:|------------:|-----------:|-----------------------:|
| pr_auc_delta | 0.000201528 | -0.0209754  | 0.0213548  |                  0.511 |
| f1_delta     | 8.23461e-05 | -0.00372376 | 0.00348602 |                  0.529 |

## Teacher adaptation

| model            |   samples |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |   specificity | confusion_matrix         |   positive_prevalence |   predicted_positive_rate |
|:-----------------|----------:|------------:|-----------:|------------:|---------:|---------:|----------:|---------:|--------------:|:-------------------------|----------------------:|--------------------------:|
| teacher_original |      2000 |       0.365 |     0.8535 |    0.843874 | 0.863498 | 0.853573 |  0.93249  | 0.941853 |      0.843719 | [[853, 158], [135, 854]] |                0.4945 |                     0.506 |
| teacher_adapted  |      2000 |       0.44  |     0.8485 |    0.836935 | 0.861476 | 0.849028 |  0.930903 | 0.938848 |      0.835806 | [[845, 166], [137, 852]] |                0.4945 |                     0.509 |

## QAT-to-TFLite parity

| model                     |   qat_keras_to_tflite_mae |   maximum_absolute_error |
|:--------------------------|--------------------------:|-------------------------:|
| hard_qat_int8             |                  0.419252 |                 0.961283 |
| fixed_teacher_qat_kd_int8 |                  0.401152 |                 0.955954 |
| qkd_full_int8             |                  0.36389  |                 0.930247 |

## Acceptance gates

|                                  |   passed |
|:---------------------------------|---------:|
| pr_auc_gain_over_hard_qat        |        1 |
| f1_retained_vs_hard_qat          |        1 |
| specificity_retained_vs_hard_qat |        1 |
| float_pr_auc_retained            |        0 |
| qat_tflite_parity                |        0 |
| integer_only                     |        1 |
| accepted_desktop                 |        0 |
| eligible_for_device              |        0 |
