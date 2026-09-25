# Notebook 01 result: teacher audit and Student-120 hard-label control

## Executive result

Notebook 01 is now a valid **hard-label control** for the distillation phase. It is not yet a
distilled model: the teacher was audited and cached, but no teacher loss was used to train this
student. This distinction is important because Notebook 02 must compare response KD against this
exact control.

The Student-120 control improves substantially over the earlier Baseline-96 and Fast-80 models,
but it remains well below the Teacher-160. That gap is large enough to justify response
distillation. The model is ready for the response-KD experiment, but it is **not yet approved for
INT8 deployment** because physical device profiling is missing and the strict float/INT8
probability-parity gate failed.

## Corrected notebook outputs

The original saved notebook did not display the main quality figures because its evaluation cell
had no execution result. The repaired evaluated notebook now embeds:

- float and INT8 confusion matrices;
- float and INT8 precision-recall curves;
- an all-model quality comparison;
- recall by person-size slice;
- accuracy by brightness quartile;
- float-versus-INT8 parity and resource summaries.

The export was also corrected. Random augmentation had leaked training-only TensorFlow variable
operators into the first TFLite graph. The notebook now rebuilds an inference-only graph, copies
the learned weights, checks float graph parity, and converts that clean graph. Its operator set is
`ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `LOGISTIC`, `MEAN`, `MUL`, and `PAD`.

## Test-set quality

All rows use the same 2,000 frozen test images. Each threshold was selected on validation data,
not on the test set.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher-160 float | 85.15% | 85.92% | 83.38% | 86.85% | 84.64% | 92.15% | 93.05% |
| Student-120 float | 73.80% | 70.31% | 80.63% | 67.22% | 75.12% | 83.18% | 84.06% |
| Student-120 INT8 PTQ | 73.60% | 69.38% | 82.67% | 64.87% | 75.44% | 82.72% | 83.53% |
| Baseline-96 INT8 | 69.10% | 64.49% | 82.36% | 56.33% | 72.34% | 80.09% | 80.34% |
| Fast-80 INT8 | 65.75% | 60.10% | 89.81% | 42.59% | 72.01% | 78.82% | 78.73% |

### Improvement over the earlier versions

Student-120 INT8 versus Baseline-96 INT8:

- accuracy: **+4.50 percentage points**;
- precision: **+4.89 points**;
- recall: **+0.31 points**;
- specificity: **+8.54 points**;
- F1: **+3.11 points**;
- PR-AUC: **+3.18 points**.

Student-120 INT8 versus Fast-80 INT8:

- accuracy: **+7.85 percentage points**;
- precision: **+9.28 points**;
- specificity: **+22.28 points**;
- F1: **+3.44 points**;
- PR-AUC: **+4.80 points**;
- recall: **-7.14 points**.

Fast-80's higher recall is not a better balanced result: it predicts `person` for 73.3% of the
test set even though person prevalence is 49.05%. Student-120 INT8 reduces that predicted-positive
rate to 58.45%, explaining the large specificity improvement.

## Confusion-matrix interpretation

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Student-120 float | 685 | 334 | 190 | 791 |
| Student-120 INT8 | 661 | 358 | 170 | 811 |

At separately calibrated validation thresholds, PTQ trades 24 true negatives for 20 additional
true positives. It therefore raises recall while lowering specificity. The INT8 F1 is 0.32 points
higher than float, but PR-AUC is 0.54 points lower, so quantization did not improve the underlying
ranking quality.

## Teacher headroom for distillation

The teacher remains much stronger than the hard-label student:

- Student-120 float is **11.35 accuracy points**, **9.52 F1 points**, and **8.99 PR-AUC points**
  below Teacher-160;
- the teacher is correct on 310 examples that the student misses;
- the student is correct on 83 examples that the teacher misses;
- the mean absolute teacher/student probability difference is 0.165.

The net 227-example teacher advantage is useful supervision headroom. Hard labels must remain in
the loss because the student still fixes 83 teacher errors.

## Error slices

Person-size recall is the clearest remaining weakness:

| Person size | Positive samples | Float recall | INT8 recall |
|---|---:|---:|---:|
| Small | 132 | 53.79% | 59.85% |
| Medium | 220 | 76.82% | 80.45% |
| Large | 629 | 87.60% | 88.24% |

The 28-point gap between small- and large-person INT8 recall is much larger than the variation
across brightness quartiles. Float accuracy varies from 72.6% to 76.4% across brightness bands,
while INT8 varies from 72.0% to 75.0%. In this test set, object scale/resolution is therefore a
stronger measured problem than global brightness.

## Float-to-INT8 parity

- probability MAE: **0.03631**, above the strict 0.03 gate;
- F1 delta: **+0.00323**;
- PR-AUC delta: **-0.00536**;
- the export is integer-only and contains no augmentation/RNG operators.

The small metric loss is acceptable for continuing research, but the probability shift means PTQ
is not yet deployment-approved. QAT should be evaluated after selecting the best float distilled
student rather than being mixed into Notebook 02.

## Static efficiency comparison

The first table contains graph estimates. The second table contains physical TFLite Micro
measurements; these evidence types must not be mixed.

| Model | Parameters | MACs / inference | Estimated peak INT8 live activation | TFLite bytes |
|---|---:|---:|---:|---:|
| Student-120 | 218,801 | 10,487,648 | 117,136 | 304,296 |
| Fast-80 | 111,793 | 3,993,536 | 51,200 | 167,976 |

Relative to Fast-80, Student-120 has about 1.96x the parameters, 2.63x the MACs, 2.29x the
estimated peak activation, and 1.81x the flatbuffer size. The quality gain is real, but it is not
free.

### Physical ESP32-CAM result

The clean Student-120 INT8 model was compiled with ESP-NN 1.3.2, flashed to the connected
ESP32-CAM, and profiled over 20 measured batch-one invocations after two warm-ups. GPIO4 was
forced low with `VWW_FLASH_LED_ENABLED=0` throughout the run.

| Metric | Fast-80 | Student-120 | Student / Fast-80 |
|---|---:|---:|---:|
| Invoke mean | 455.546 ms | 1,117.123 ms | 2.45x |
| Invoke range | 453.364-459.697 ms | 1,114.846-1,121.736 ms | — |
| Measured TFLM arena used | 69,068 B | 142,444 B | 2.06x |
| TFLite flatbuffer | 167,976 B | 304,296 B | 1.81x |
| Effective throughput | 8.77 MMAC/s | 9.39 MMAC/s | 1.07x |
| Full camera-loop FPS | not recorded in original profile | 0.74 FPS | — |
| Student preprocessing median | — | 95.5 ms | — |
| Student active pipeline median | — | 1,256.1 ms | — |

The similar effective MMAC/s shows that the slowdown is primarily additional graph work rather
than a new kernel-efficiency collapse. Student-120 is deployable, but it is not an ESP32-CAM
latency Pareto winner. Its quality gain costs approximately 662 ms more Invoke time and 73,376 B
more measured arena than Fast-80.

### Physical ESP32-S3 result

The same `student_120_hard_control_int8.tflite` artifact (SHA-256
`99b30524db595e90a939274bfac66977694375657bbaeb84217fcdf2e9b13dc0`) was flashed to the
connected ESP32-S3 revision 0.2 and measured at batch one. The board reports a 240 MHz dual-core
CPU, 16 MiB DIO flash at 80 MHz, and 8 MiB octal PSRAM at 80 MHz. Camera, Wi-Fi, and LEDs were not
initialized by the profiling firmware.

| Metric | Fast-80 internal | Student-120 internal | Fast-80 PSRAM | Student-120 PSRAM |
|---|---:|---:|---:|---:|
| Invoke mean | 67.105 ms | 150.906 ms | 73.478 ms | 176.005 ms |
| Invoke p95 | 67.119 ms | 150.923 ms | 73.494 ms | 176.023 ms |
| Model-only FPS | 14.90 | 6.63 | 13.61 | 5.68 |
| Measured TFLM arena used | 83,420 B | 159,212 B | 83,420 B | 159,212 B |
| Effective throughput | 59.51 MMAC/s | 69.50 MMAC/s | 54.35 MMAC/s | 59.59 MMAC/s |

Student-120 is 2.25x slower than Fast-80 with the arena in internal SRAM and 2.40x slower with the
arena in PSRAM. Internal placement reduces Student-120 latency by 14.26% versus PSRAM. More
importantly, the S3 executes the exact Student-120 graph **7.40x faster than ESP32-CAM** (150.906 ms
versus 1,117.123 ms), while using 159,212 B of arena. The S3 therefore makes Student-120 practical
for responsive model-only inference, although end-to-end camera FPS still requires a separate
camera-pipeline measurement.

## Training behavior

Validation PR-AUC increased throughout the two-stage run and reached approximately 0.846 near the
last epoch. Test PR-AUC was 0.8406, close to validation, with no sign of a severe generalization
collapse. Lower reported training metrics are expected because camera augmentation is active only
during training.

## Decision

- **Ready for response KD:** yes.
- **Deployable on ESP32-CAM:** yes; the clean integer graph allocates and invokes successfully.
- **Ready to promote as the ESP32-CAM runtime model:** no; 1,117 ms Invoke and 0.74 pipeline FPS
  are substantial regressions versus Fast-80.
- **ESP32-S3 deployment complete:** yes; the exact Student-120 INT8 artifact runs successfully in
  both internal-SRAM and PSRAM arena placements.
- **Preferred S3 placement:** internal SRAM, because it is 14.26% faster and the 159,212 B used arena
  fits the measured contiguous internal-memory capacity.
- **Next controlled experiment:** Notebook 02, response KD with the same Student-120 architecture,
  initialization/training budget, data split, preprocessing, and export contract. Start with the
  MicroNets-derived anchor `T=4`, `lambda=0.5`, then run the registered temperature/weight
  ablations.
