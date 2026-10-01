# Notebook 05 outcome report: FitNets feature hints

## Executive decision

Notebook 05 is a **successful FitNets mechanism experiment but not a model-promotion result**.
The paper-derived hint stage learned the teacher's intermediate representation, the subsequent
student improved decisively over the notebook's random-initialized KD control, and full-INT8
conversion preserved its quality. However, the final student remained materially worse than the
project's stronger ImageNet-initialized Student-120 reference from Notebook 03. We therefore pass
on further FitNets work in the current distillation sequence and do not deploy this model.

| Decision level | Outcome | Evidence |
|---|---|---|
| FitNets implementation | Pass | Hint CKA rose from 0.3266 to 0.9562 at the Stage-1 checkpoint |
| Causal comparison inside Notebook 05 | Pass | PR-AUC +0.1557 and F1 +0.0622 versus its KD control; both bootstrap intervals exclude zero |
| Deployable graph | Pass | 218,801 parameters and 10.49M MACs for every branch; the training-only regressor is absent |
| Full-INT8 parity | Pass | Probability MAE 0.0141; F1 changed by -0.00007 and PR-AUC by +0.00071 |
| Project champion comparison | **Fail** | PR-AUC 0.7985 versus 0.8651 and F1 0.7312 versus 0.7772 |
| Physical deployment | Not run | The candidate failed the project-quality gate before board profiling |

## Question and experimental contract

The notebook asked whether the two-stage FitNets procedure can improve a thin Student-120 when
the student begins from random initialization:

1. **Hint stage:** train a temporary `1x1` regressor so the student's `conv_pw_6_relu` feature
   predicts the teacher's `conv_pw_6_relu` feature.
2. **Guided stage:** discard the regressor and optimize the student with hard VWW labels plus
   softened teacher responses.

The teacher is MobileNetV1, width multiplier 0.50, with `160x160x3` RGB input. The student is
MobileNetV1, width multiplier 0.25, with `120x120x3` RGB input. Both branches receive the same
semantic augmented view before their independent resize operations. The teacher remains frozen.

The compared Stage-2 branches share the same random student initialization:

| Branch | Stage 1 | Stage 2 objective | Purpose |
|---|---|---|---|
| `hard_control` | None | Hard-label BCE | Random-initialized hard-label reference |
| `kd_control` | None | Hard BCE + response KD | Controls for Stage-2 teacher supervision |
| `fitnets_hint_kd` | FitNets hint regression | Same hard BCE + response KD | Measures the value of hint pretraining |

This is a valid internal causal comparison. It is not equivalent to the repository's best
Student-120 training recipe because Notebook 03 uses ImageNet initialization and a stronger
fine-tuning protocol.

## Paper-derived method

The notebook implements the two-stage idea from *FitNets: Hints for Thin Deep Nets*:

\[
L_{hint}=\frac{1}{N}\left\|F_T-r(F_S)\right\|_2^2,
\]

where `r` is a temporary learned regressor. The selected layers have comparable network roles and
effective reductions, while their spatial and channel dimensions differ:

| Network | Layer | Shape | Effective spatial reduction |
|---|---|---:|---:|
| Teacher | `conv_pw_6_relu` | `10x10x256` | 16.00 |
| Student | `conv_pw_6_relu` | `7x7x128` | 17.14 |

The implementation resizes the student feature spatially and uses a train-only `1x1` convolution
to match teacher channels. That regressor is used only for Stage 1 and diagnostics; it is not part
of the exported student.

Stage 2 combines hard binary cross-entropy and temperature-softened teacher supervision with
`T=3`. Its teacher-loss weight decays from 4 to 1. This follows the experiment's registered
configuration, although it is not a claim that these are universally optimal values.

## Training audit

The completed run trained 72 epochs across four stages and took about 3.87 hours.

| Stage | Epochs | Approx. time | Key observation |
|---|---:|---:|---|
| Hint regression | 12 | 24.5 min | Validation hint loss fell from 0.6888 to 0.3071 |
| Hard control | 20 | 87.4 min | Best validation PR-AUC 0.5904 |
| KD control | 20 | 51.7 min | Best validation PR-AUC 0.6239 |
| FitNets + KD | 20 | 68.8 min | Best/final validation PR-AUC 0.7891 |
| **Total** | **72** | **232.4 min** | About 27,000 optimizer updates |

The hard-control runtime includes unusually long epochs 15–17. This is consistent with host sleep,
thermal throttling, or resource contention rather than a graph change and should not be interpreted
as model latency.

The mean hard/soft gradient cosine was about 0.672 for the KD control and 0.606 for FitNets. The
positive values show broadly compatible objectives; the lower FitNets value also shows that the
teacher response was not simply duplicating the hard-label gradient.

## Representation-transfer evidence

| Checkpoint | Linear CKA to teacher | Hint MSE through Stage-1 regressor |
|---|---:|---:|
| Shared random initialization | 0.3266 | 4.2031 |
| End of hint stage | **0.9562** | **0.6108** |
| KD control final | 0.6172 | 1.5866 |
| FitNets final | **0.9177** | **0.8429** |

Stage 1 reduced hint MSE by about **85.5%** and raised CKA by 0.6296. Most of this alignment
persisted through Stage 2. The KD control also moved toward the teacher, but it remained much less
aligned. This is the clearest evidence that the hint mechanism itself worked.

CKA and feature MSE are diagnostics rather than deployment objectives. They establish that
representation knowledge transferred; they do not establish that the resulting classifier is the
best project model.

## Test-set quality

Thresholds were selected on validation data and then frozen for the 2,000-image test set.

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard control | 0.395 | 0.5700 | 0.5384 | 0.8644 | 0.2866 | 0.6635 | 0.6379 | 0.6147 | 0.0165 |
| KD control | 0.365 | 0.5450 | 0.5201 | 0.9368 | 0.1678 | 0.6689 | 0.6628 | 0.6426 | 0.0305 |
| FitNets + KD | 0.180 | **0.6985** | **0.6498** | 0.8359 | **0.5662** | **0.7312** | **0.8031** | **0.7985** | 0.1568 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

| Model | TN | FP | FN | TP | Predicted-person rate |
|---|---:|---:|---:|---:|---:|
| Hard control | 292 | 727 | 133 | 848 | 78.75% |
| KD control | 171 | 848 | 62 | 919 | 88.35% |
| FitNets + KD | 577 | 442 | 161 | 820 | 63.10% |

FitNets corrected much of the controls' severe person-class bias. It traded some recall for 405
fewer false positives than the KD control, producing substantially better specificity and global
ranking quality.

The low operating threshold (`0.18`) and high ECE (`0.1568`) are warnings. The model's ranking is
useful, but its raw probabilities are poorly calibrated. Firmware must use the validation-selected
threshold rather than `0.5`; deployment would also require camera-domain recalibration.

## Paired uncertainty analysis

The paired bootstrap compares FitNets + KD with the KD control on identical test examples.

| Metric delta | Mean | 95% interval | Probability delta > 0 |
|---|---:|---:|---:|
| PR-AUC | +0.1557 | [0.1274, 0.1858] | 1.000 |
| F1 | +0.0622 | [0.0457, 0.0811] | 1.000 |

These intervals support a real improvement over the notebook's internal control. They do not
repair the weaker initialization contract or make the result competitive with Notebook 03.

## Comparison with accepted project references

| Model | Accuracy | F1 | PR-AUC | Specificity | ECE |
|---|---:|---:|---:|---:|---:|
| Teacher-160 | 0.8515 | 0.8464 | 0.9305 | 0.8685 | — |
| Notebook 03 `same_view_lambda0` | **0.7815** | **0.7772** | **0.8651** | **0.7861** | 0.0394 |
| Notebook 04 attention control | 0.7730 | 0.7766 | 0.8633 | 0.7429 | 0.0166 |
| Notebook 05 FitNets + KD | 0.6985 | 0.7312 | 0.7985 | 0.5662 | 0.1568 |

Relative to the Notebook 03 champion, FitNets loses:

- 8.30 percentage points of accuracy;
- 4.60 points of F1;
- 6.66 points of PR-AUC;
- 21.98 points of specificity;
- and adds 11.74 points of calibration error.

The dominant explanation is the experiment contract: Notebook 05 intentionally starts from
random weights to test classic FitNets, whereas the project already has a much stronger
ImageNet-initialized student recipe. A pretrained-FitNets follow-up could answer a different
question, but it is not justified now because the user elected to pass this technique.

## Efficiency and exported graph

Distillation changes training, not the deployable architecture. All float branches have the same
static graph:

| Resource | Value | Evidence type |
|---|---:|---|
| Parameters | 218,801 | Static model inspection |
| Estimated MACs, batch 1 | 10,487,648 | Static graph estimate |
| Float32 weights | 875,204 bytes | Static estimate |
| Peak live float activation | 468,544 bytes | Static liveness estimate |
| Peak live INT8 activation | 117,136 bytes | Hypothetical static estimate |
| Peak layer | `conv_pad_2` | Static liveness estimate |

The activation estimate is **not** an ESP32 tensor-arena or SRAM measurement. It excludes kernel
workspace, allocator fragmentation, runtime objects, capture buffers, and operator-specific
scratch memory.

The full-INT8 artifact is 304,944 bytes and uses only `ADD`, `CONV_2D`,
`DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `LOGISTIC`, `MEAN`, `MUL`, and `PAD`. Input and output are
both `int8`. The input shape is `1x120x120x3`; the output shape is `1x1`.

| INT8 metric | Result |
|---|---:|
| Threshold | 0.175 |
| Accuracy | 0.6925 |
| Precision | 0.6401 |
| Recall | 0.8522 |
| Specificity | 0.5388 |
| F1 | 0.7311 |
| PR-AUC | 0.7992 |
| Float-to-INT8 probability MAE | 0.0141 |

Quantization preserved the float model extremely well. It did not make the model better than the
Notebook 03 reference, and it does not imply that the model fits an ESP32-CAM tensor arena.

## Acceptance gates and corrected interpretation

The notebook's local gates all passed because they compare the candidate with the random-initialized
Notebook 05 controls:

- PR-AUC gain: pass;
- F1 retention: pass;
- specificity retention: pass;
- feature-alignment gain: pass;
- graph identity: pass;
- local float acceptance: pass.

For project decisions, a second gate is required: compare with the best prior student under the
same test contract. That gate fails. Future notebooks must report both **internal causal gates**
and the **project champion gate** so that a weak control cannot promote a globally inferior model.

## Reproducibility caveat

Five code cells in the saved notebook have no execution count even though later dependent cells
contain outputs. This most likely means the notebook was edited or executed out of order after
training. The persisted artifacts are internally coherent and contain configuration, manifest,
teacher, and paper hashes, but the notebook file is not a clean top-to-bottom execution record.

If this experiment ever needs formal certification, restart the kernel, set checkpoint reuse on,
and run all cells once. That should rebuild the report without repeating the expensive training.

## Final conclusion

FitNets did what it was designed to do: it forced a randomly initialized thin student into a much
closer intermediate representation and produced a statistically clear classification improvement.
The experiment therefore validates the implementation and the core paper mechanism. It does not
produce the best VWW student. The correct action is to retain this notebook and report as research
evidence, skip physical deployment, and proceed to a relation-based technique using the strong
ImageNet-initialized Student-120 contract.

## Artifact index

- Notebook: `05_fitnets_feature_hints.ipynb`
- Configuration: `configs/distillation_fitnets_120.yaml`
- Compact experiment record: `artifacts/distillation/fitnets_student_120/reports/experiment_summary.json`
- Quality table: `artifacts/distillation/fitnets_student_120/reports/quality_comparison.csv`
- Feature diagnostics: `artifacts/distillation/fitnets_student_120/reports/feature_diagnostics.csv`
- Bootstrap summary: `artifacts/distillation/fitnets_student_120/reports/bootstrap_summary.csv`
- Graph comparison: `artifacts/distillation/fitnets_student_120/reports/graph_identity.csv`
- INT8 parity: `artifacts/distillation/fitnets_student_120/reports/float_to_int8_parity.json`
- Figures: `artifacts/distillation/fitnets_student_120/figures/`
