# Notebook 07 outcome report: Quantization-aware Knowledge Distillation

**Experiment:** `vww-qkd-int8-student-120`  
**Audit date:** 2026-10-05  
**Decision:** **Do not deploy the QAT/QKD TFLite artifacts. Keep the PTQ INT8 model as the current deployable candidate while the QAT export path is repaired.**

## Executive assessment

Notebook 07 completed all training branches and generated the expected checkpoints, reports, figures, and integer-only TFLite files. The training run itself did not collapse. Reconstructed QAT checkpoints achieve approximately **0.866 test PR-AUC**, essentially retaining the float student's **0.8651 PR-AUC**.

The failure occurs after conversion to TFLite. All three QAT-derived INT8 exports lose class separation and fall to approximately **0.522-0.526 PR-AUC**, close to random ranking on this balanced test set. QAT-to-TFLite probability MAE is **0.364-0.419**, more than twelve times the configured maximum of 0.03. Consequently, the final QKD artifact fails the desktop acceptance gate and is not eligible for ESP32 profiling or deployment.

This experiment therefore does **not** yet answer whether QKD improves an ESP32 deployment. It shows that:

1. the QAT checkpoints learned useful classifiers;
2. full QKD produced no material quality gain over hard-label QAT in the Keras fake-quantized graph; and
3. the current QAT-to-TFLite conversion graph is invalid for deployment parity.

## Experiment contract

| Item | Value |
|---|---:|
| Train / validation / test | 12,000 / 2,000 / 2,000 |
| Test person prevalence | 49.05% |
| Teacher input | 160 x 160 x 3 RGB |
| Student input | 120 x 120 x 3 RGB |
| Student architecture | MobileNetV1, alpha 0.25 |
| Parameters | 218,801 |
| Estimated MACs, batch 1 | 10,487,648 |
| QAT format | W8A8 |
| QKD temperature | 2.0 |
| KD weight | 1.0 |
| Full QKD schedule | 8 SS + 6 CS + 6 TU = 20 epochs |
| Runtime | TensorFlow 2.20.0, tf-keras 2.20.1, TFMOT 0.8.1 |

The paper-derived schedule is represented as:

- **Self-studying (SS):** student learns from hard labels with fake quantization active.
- **Co-studying (CS):** student and partially unfrozen teacher exchange Bernoulli-KL supervision.
- **Tutoring (TU):** the adapted teacher is frozen and supervises the quantized student.

The controls are reasonably budget-aligned: hard-label QAT and fixed-teacher QAT+KD both continue for 12 epochs after the shared 8-epoch SS checkpoint, giving each path 20 student epochs.

## Training behavior

| Branch | Epochs in log | Best validation PR-AUC | Final validation PR-AUC | Runtime |
|---|---:|---:|---:|---:|
| Self-study | 8 | 0.85685 | 0.85685 | 8.01 min |
| Hard-label QAT continuation | 12 | **0.86843** | 0.86580 | 14.93 min |
| Fixed-teacher QAT+KD | 12 | 0.86838 | 0.86476 | 21.56 min |
| Co-study | 6 | 0.86597 | 0.86468 | 12.75 min |
| Tutoring | 6 | 0.86598 | 0.86598 | 9.44 min |

The full QKD path took approximately **30.20 minutes** after combining SS, CS, and TU. Running all branches cost approximately **66.69 minutes**.

The hard-label QAT control achieved the highest validation PR-AUC. Fixed-teacher KD was statistically indistinguishable at its best epoch, while the paper's full SS-CS-TU path did not exceed either control. Training curves remain stable; there is no NaN, exploding loss, or validation collapse.

## Quality before and after TFLite conversion

### Saved QAT checkpoints evaluated in Keras

These values were reconstructed directly from the saved checkpoints using the frozen validation threshold procedure and the same 2,000-image test split.

| Checkpoint | Threshold | Accuracy | Recall | Specificity | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Float champion | 0.385 | 0.7815 | 0.7768 | 0.7861 | 0.7772 | 0.86510 |
| Hard QAT | 0.425 | 0.7795 | 0.7798 | 0.7792 | 0.7763 | **0.86607** |
| Fixed-teacher QAT+KD | 0.400 | 0.7760 | 0.7961 | 0.7566 | 0.7771 | 0.86556 |
| Full QKD | 0.385 | 0.7755 | **0.8033** | 0.7488 | **0.7783** | 0.86565 |

Interpretation:

- Full QKD retains float quality in the fake-quantized Keras graph: PR-AUC is +0.00055 and F1 is +0.00112 relative to the float champion.
- Against the fair hard-QAT control, full QKD changes PR-AUC by **-0.00042**, F1 by **+0.00202**, accuracy by **-0.0040**, and specificity by **-0.03042**.
- The small F1 gain is obtained through a more recall-oriented operating point, not better ranking. Full QKD raises recall by 2.34 percentage points but loses 3.04 points of specificity against hard QAT.
- There is no evidence that KD materially improves this student under the present schedule.

### Exported TFLite INT8 artifacts

| Artifact | Threshold | Accuracy | Recall | Specificity | F1 | PR-AUC | Predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Float champion | 0.385 | 0.7815 | 0.7768 | 0.7861 | 0.7772 | 0.86510 | 49.00% |
| PTQ INT8 | 0.275 | 0.7615 | 0.7798 | 0.7439 | 0.7623 | **0.85685** | 51.30% |
| Hard QAT INT8 | 0.010 | 0.4905 | 1.0000 | 0.0000 | 0.6582 | 0.52221 | 100.00% |
| Fixed-teacher QAT+KD INT8 | 0.025 | 0.4975 | 0.9959 | 0.0177 | 0.6604 | 0.52573 | 98.90% |
| Full QKD INT8 | 0.050 | 0.4960 | 0.9898 | 0.0206 | 0.6583 | 0.52234 | 98.45% |

PTQ remains a credible deployment candidate: it loses 0.00825 PR-AUC and 0.01482 F1 from float, but still preserves useful class separation. Every QAT-derived TFLite model collapses toward the positive decision.

The exported QKD probabilities are not intrinsically high; they are compressed into **0.0469-0.3750** with a mean of **0.1158**. Validation then selects the unusually low threshold 0.05, placing 98.45% of test images above it. This explains the apparent “always person” behavior.

## Conversion-parity diagnosis

| Model | Keras-to-TFLite MAE | Maximum absolute error | Pearson correlation |
|---|---:|---:|---:|
| Hard QAT | 0.41925 | 0.96128 | 0.0982 |
| Fixed-teacher QAT+KD | 0.40115 | 0.95595 | 0.0792 |
| Full QKD | 0.36389 | 0.93025 | 0.0582 |

The configured parity limit is 0.03. These are not ordinary INT8 rounding errors: prediction ordering is almost destroyed.

### Most likely root cause

The reconstructed QAT graph has the repeated structure:

```text
QuantizeWrapper(Conv/DepthwiseConv) -> unwrapped BatchNormalization -> QuantizeWrapper(ReLU)
```

The fake-quantizer is therefore trained around the convolution **before** batch normalization. During TFLite conversion, batch normalization is folded into the preceding convolution, so the deployed integer convolution has different effective weights and activation ranges from the operation simulated during QAT. This mismatch is especially severe here because at least one saved BN moving variance is approximately `1.9e-26`; folding such a BN can strongly amplify the effective convolution scale.

That graph evidence, combined with healthy Keras-QAT quality, catastrophic parity, low Keras/TFLite correlation, and a working PTQ conversion of the same float architecture, makes Conv-BN folding/range mismatch the leading diagnosis. A first-divergent-layer parity trace should still be used to prove the exact layer before changing the training recipe.

This is not caused by the RGB dataset, the 120 x 120 resolution, the decision threshold, or the ESP32 camera. The failure is reproducible on the desktop before hardware is involved.

## Statistical comparison

The notebook's paired bootstrap compares full-QKD INT8 with hard-QAT INT8:

| Delta, QKD minus hard QAT | Mean | 95% CI | Probability positive |
|---|---:|---:|---:|
| PR-AUC | +0.00020 | [-0.02098, +0.02135] | 0.511 |
| F1 | +0.00008 | [-0.00372, +0.00349] | 0.529 |

This shows no significant difference, but both inputs to the comparison are already broken exports. The result cannot be used to validate QKD. The bootstrap must be repeated on repaired INT8 artifacts and should also include a Keras-checkpoint comparison.

## Teacher adaptation

| Teacher | Accuracy | F1 | PR-AUC | Specificity |
|---|---:|---:|---:|---:|
| Original | 0.8535 | 0.85357 | 0.94185 | 0.84372 |
| Adapted after co-study | 0.8485 | 0.84903 | 0.93885 | 0.83581 |
| Delta | -0.0050 | -0.00455 | -0.00301 | -0.00791 |

Co-study slightly degraded the teacher. That does not explain the export collapse, but it does explain why the full paper path did not beat the fixed-teacher or hard-QAT controls. If QKD is retrained later, reduce or eliminate teacher updates unless an ablation shows a repeatable gain.

## Efficiency and hardware implications

| Artifact | FlatBuffer | Desktop median | Params | MACs | Estimated peak INT8 activation |
|---|---:|---:|---:|---:|---:|
| PTQ INT8 | 303,304 B (296.2 KiB) | 0.1955 ms | 218,801 | 10.49 M | 160,336 B (156.6 KiB) |
| Hard QAT INT8 | 320,952 B (313.4 KiB) | 0.3631 ms | 218,801 | 10.49 M | 160,336 B |
| Fixed-teacher QAT+KD INT8 | 320,952 B | 0.3624 ms | 218,801 | 10.49 M | 160,336 B |
| Full QKD INT8 | 320,952 B | 0.3625 ms | 218,801 | 10.49 M | 160,336 B |

QKD changes training, not architecture, so it does not reduce parameter count, MACs, or the static activation estimate. The current QAT FlatBuffers are 17,648 bytes (**5.82%**) larger than PTQ, and their desktop median latency is about **85%** higher. Desktop latency is only a conversion sanity check and must not be presented as ESP32 latency.

No physical device metrics were collected for this run. Tensor-arena size, peak internal SRAM, PSRAM use, camera preprocessing time, invoke latency, end-to-end FPS, and energy remain unknown. Hardware profiling is correctly blocked until desktop parity passes.

## Acceptance-gate interpretation

The experiment correctly reports:

- `float_pr_auc_retained = false`
- `qat_tflite_parity = false`
- `accepted_desktop = false`
- `eligible_for_device = false`

Three relative gates pass because full QKD is marginally similar to hard QAT after both exports collapse. Those passes are not evidence of success. The authoritative blockers are float retention and Keras-to-TFLite parity.

## Reproducibility note

The notebook contains two required definition cells—teacher co-study configuration and callbacks—with no execution count, while dependent cells are executed. The artifacts prove those definitions existed in the live kernel, but the saved notebook is not a clean top-to-bottom execution record. After the export fix, run **Restart Kernel and Run All** so the execution order is auditable.

## Recommended next experiment

Do not retrain the 20-epoch QKD schedule yet. Repair and validate the export path first:

1. Build a deployment-equivalent float model with Conv-BN folding handled before QAT, or use a supported QAT transformation that treats Conv-BN-activation as one deployment pattern.
2. Add per-layer parity on a fixed calibration batch and identify the first tensor whose Keras and TFLite values diverge.
3. Export the existing hard-label QAT control first. Require probability MAE <= 0.03, high prediction correlation, and PR-AUC close to its Keras checkpoint before touching QKD.
4. Only after hard-QAT export passes, apply the same corrected graph to fixed-teacher KD and full QKD.
5. Compare repaired artifacts against PTQ. Continue QKD only if the gain exceeds uncertainty and the artifact does not regress size, arena use, or measured ESP32 latency.
6. Profile the accepted model on ESP32-S3 first, then ESP32-CAM, with flash disabled during profiling.

The immediate engineering target is therefore **QAT export parity**, not more QKD training.

## Evidence locations

- Notebook: `07_quantization_aware_kd.ipynb`
- Generated notebook summary: `artifacts/distillation/qkd_int8_student_120/reports/QKD_REPORT.md`
- Reconstructed pre-export metrics: `artifacts/distillation/qkd_int8_student_120/reports/qat_keras_quality_diagnostic.csv`
- Training histories: `artifacts/distillation/qkd_int8_student_120/reports/training_history.csv`
- TFLite quality: `artifacts/distillation/qkd_int8_student_120/reports/quality_comparison.csv`
- Parity: `artifacts/distillation/qkd_int8_student_120/reports/qat_to_tflite_parity.csv`
- Hardware summary: `artifacts/distillation/qkd_int8_student_120/reports/hardware_summary.csv`
- Acceptance gates: `artifacts/distillation/qkd_int8_student_120/reports/acceptance_gates.json`
