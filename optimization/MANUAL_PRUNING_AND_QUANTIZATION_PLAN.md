# ESP32-CAM VWW manual pruning and quantization plan

## Purpose

This document defines the next optimization phase for the deployed 80x80 Visual Wake Words
(VWW) model. The goal is to improve physical-device latency and memory use without changing
the task, training dataset, or batch size.

The work starts from the frozen `fast_80` full-INT8 model. Every candidate must be compared
with that exact reference before it can replace the deployed model. This document proposes
experiments; it does not apply pruning, retraining, or a new quantization scheme.

## Fixed project constraints

- Task: binary `person` / `no person` Visual Wake Word classification.
- Dataset: the existing MS COCO-derived VWW train, validation, and held-out test splits.
- No replacement or additional training dataset.
- Device: AI-Thinker ESP32-CAM form factor with ESP32-D0WD-V3 and OV3660 sensor.
- Runtime: TensorFlow Lite Micro with ESP-NN, batch size exactly one.
- Camera path: JPEG capture, RGB decode, fixed-point camera-domain transform, resize, and
  model-input quantization.
- Deployment output: a full-INT8 TFLite FlatBuffer with no float fallback.
- Current device policy remains separate from the model: threshold `0.44`, motion threshold
  `2.0`, and 2-of-3 temporal voting.

Captured device frames may be used for unlabeled calibration, preprocessing parity, and
hardware validation if needed, but they do not replace the fixed training or test datasets.

## Frozen reference: fast 80

| Perspective | Reference value |
|---|---:|
| Input | `1x80x80x3` INT8 |
| Parameters | 111,793 |
| TFLite model | 167,976 B |
| Estimated MACs | 3,993,536 / inference |
| Fused TFLite operators | 26 |
| Fused-TFLite live activation peak | 38,400 B |
| Actual TFLM arena use | 69,068 B |
| Arena reservation | 358,400 B in mapped PSRAM |
| Controlled instrumented Invoke mean | 455.546 ms |
| Latest firmware Invoke mean | 416.8 ms |
| Latest preprocessing | 59.8-61.3 ms |
| Latest complete frame rate | 1.66-1.68 fps, including 100 ms delay |
| INT8 test accuracy | 65.75% |
| INT8 test precision | 60.10% |
| INT8 test recall | 89.81% |
| INT8 test specificity | 42.59% |
| INT8 test F1 | 72.01% |
| INT8 test ROC-AUC / PR-AUC | 78.82% / 78.73% |
| Offline validation-selected threshold | 0.27 |

The controlled latency and latest firmware latency are different experiments. Controlled
per-operator measurements are used for version-to-version attribution. Production firmware
timing is used for end-to-end behavior and FPS.

## Hardware implications

| Observation | Optimization implication |
|---|---|
| No dedicated ML accelerator | The graph must use efficient dense INT8 CPU kernels. |
| `CONV_2D` uses 343.1 ms, 75.3% of profiled Invoke | Pointwise channel reduction is the primary pruning target. |
| Depthwise convolutions use 70.2 ms, 15.4% | Depthwise work matters, but less than pointwise work. |
| Graph normalization uses 37.6 ms, 8.2% | Move normalization into firmware input preparation. |
| Mapped PSRAM copy throughput is 4.71 MiB/s | Activation traffic and arena location are important. |
| Internal SRAM copy throughput is 181.55 MiB/s | A safely smaller arena may unlock a faster memory placement. |
| Largest ready-state internal block is 61,440 B | The current 69,068-byte arena cannot fit internally. |
| Flash partition has about 1.89 MB free | Weight storage is not the limiting resource. |
| Camera-only capture reaches 6.94 fps | Camera capture is not the current throughput bottleneck. |

The optimization objective is therefore not maximum sparsity or minimum model-file size. It
is minimum measured latency subject to accuracy and runtime-memory constraints.

## Current graph diagnosis

### Operator families

| Operator family | Count | Mean latency | Invoke share | MACs | MAC share |
|---|---:|---:|---:|---:|---:|
| `CONV_2D` | 11 | 343.112 ms | 75.3% | 3,507,712 | 87.8% |
| `DEPTHWISE_CONV_2D` | 10 | 70.160 ms | 15.4% | 485,568 | 12.2% |
| `ADD` | 1 | 24.752 ms | 5.4% | 0 | 0% |
| `MUL` | 1 | 12.878 ms | 2.8% | 0 | 0% |
| `MEAN` | 1 | 3.124 ms | 0.7% | 0 | 0% |
| `LOGISTIC` | 1 | 0.320 ms | 0.1% | 0 | 0% |
| `FULLY_CONNECTED` | 1 | 0.212 ms | less than 0.1% | 256 | less than 0.1% |

The dense classifier is too small to be an optimization target. Most useful work must occur
inside the pointwise convolutions and early activation path.

### Current channel schedule

The pointwise output channels are:

```text
stem, block1, block2, block3, block4, block5,
block6, block7, block8, block9, block10

8, 16, 32, 32, 64, 64, 128, 128, 128, 128, 256
```

All pruning candidates should initially keep channel counts in multiples of eight. This
avoids mixing architectural improvements with possible kernel tail-processing penalties.

### High-value operators

| Operator | Shape | Mean latency | MACs | Interpretation |
|---:|---|---:|---:|---|
| 6 `CONV_2D` | `20x20x16 -> 20x20x32` | 54.282 ms | 204,800 | Highest latency despite modest MAC count; high priority. |
| 2 `CONV_2D` | `80x80x3 -> 40x40x8` | 47.748 ms | 345,600 | Early, activation-heavy, and accuracy-sensitive. |
| 8 `CONV_2D` | `20x20x32 -> 20x20x32` | 31.589 ms | 409,600 | Strong compute target. |
| 12 `CONV_2D` | `10x10x64 -> 10x10x64` | 27.989 ms | 409,600 | Strong compute target. |
| 16, 18, 20 `CONV_2D` | `5x5x128 -> 5x5x128` | 81.409 ms total | 1,228,800 | Repeated late compute; suitable for channel or block pruning. |
| 22 `CONV_2D` | `3x3x128 -> 3x3x256` | 34.490 ms | 294,912 | Large constant footprint; prune cautiously. |

### Repeated-block opportunity

The three repeated 5x5, 128-channel depthwise-separable blocks are independently testable:

| Operator pair | Mean latency | MACs | Constants |
|---|---:|---:|---:|
| 15-16 | 31.010 ms | 438,400 | 18,560 B |
| 17-18 | 30.965 ms | 438,400 | 18,560 B |
| 19-20 | 31.978 ms | 438,400 | 18,560 B |

Removing one pair reduces the analytical model total to approximately 3.555M MACs. It will
reduce latency, operators, and weights, but it is not expected to reduce the early activation
peak.

## Selected pruning techniques

### 1. Complete graph-equivalent normalization removal

The exported model currently performs:

```text
raw INT8 input -> MUL -> ADD -> stem convolution
```

Firmware controls model-input conversion, so it can instead perform the normalization while
constructing the INT8 tensor. A rebuilt model can accept normalized real values directly and
begin with the stem convolution.

Implementation requirements:

1. Build an equivalent Keras inference model without the `Rescaling` layer.
2. Copy all trained convolution, BatchNorm, and classifier weights unchanged.
3. Normalize representative samples before conversion.
4. Confirm the new TFLite input scale and zero point.
5. Update firmware to use that exact input contract.
6. Compare old and new float outputs, INT8 outputs, operator inventory, and device outputs.

This is version `2.1`. It should be completed before pruning so every later candidate inherits
the simpler input graph. The measured upper-bound opportunity is approximately 37.6 ms, but
the actual saving and arena change must be measured.

### 2. Structured block pruning

Remove one whole repeated depthwise-separable block at a time. Because the candidate blocks
have identical input and output shapes, this is simpler and safer than arbitrary layer
removal.

Procedure:

1. Produce three candidates: remove operators corresponding to block 7, 8, or 9.
2. Initialize every surviving layer from `fast_80`.
3. Fine-tune each candidate on the unchanged training split.
4. Select by validation PR-AUC and the complete deployment gates.
5. Do not select only by training loss or analytical MACs.

### 3. Dependency-aware structured channel pruning

Output-filter pruning must propagate through the branch-free MobileNetV1 chain:

```text
pointwise output channel removed
    -> matching BatchNorm channel removed
    -> matching next depthwise channel removed
    -> matching input channel in the next pointwise convolution removed
```

Pruning a channel only by writing zeros is not sufficient. Tensor shapes and weight tensors
must physically become smaller so dense ESP-NN kernels execute less work.

Candidate importance signals, in recommended experiment order:

1. BatchNorm gamma magnitude.
2. L1 norm of each pointwise output filter.
3. L2 norm of each pointwise output filter.
4. First-order Taylor importance, `abs(weight * gradient)`.
5. Mean activation importance over the existing validation split.

Start with gamma and L1 ranking because they are deterministic, inexpensive, and auditable.
If their choices disagree materially, use a short sensitivity run to select the ranking rule.

### 4. Hardware-aware sensitivity analysis

For every pointwise layer, independently test channel reductions of approximately 12.5%, 25%,
and 37.5%, rounded to multiples of eight. After a short identical fine-tune, record:

- Validation and held-out accuracy metrics.
- Analytical MAC and parameter reduction.
- Fused-TFLite live activation peak.
- Actual TFLM arena use.
- Physical-device operator latency.

The decision score must value measured milliseconds and arena bytes, not just MACs. Operator 6
demonstrates why: it is slower than several operators with twice as many MACs.

## Candidate architecture frontier

The following schedules are starting hypotheses. They are not final architecture decisions.

| Candidate | Channel schedule | Estimated MACs | Reduction from fast 80 |
|---|---|---:|---:|
| Reference | `8,16,32,32,64,64,128,128,128,128,256` | 3.994M | 0% |
| Block-pruned | Reference minus one repeated 128-channel block | 3.555M | 11.0% |
| Moderate A | `8,16,24,24,48,48,104,104,104,104,208` | 2.824M | 29.3% |
| Moderate A plus block pruning | Moderate A minus one repeated block | approximately 2.531M | 36.6% |
| Moderate B | `8,16,24,32,48,64,96,96,96,96,192` | 2.901M | 27.4% |
| Aggressive | `8,16,24,24,40,48,80,80,80,80,160` | 2.302M | 42.4% |

The preferred initial 3 fps candidate is Moderate A plus one removed block. The aggressive
candidate defines a useful lower-compute boundary but carries more accuracy risk.

The estimates above include multiply-accumulates from the convolutional graph and classifier.
They do not predict exact latency because ESP32 operator throughput varies substantially by
shape, channel count, and memory traffic.

## Activation and tensor-arena strategy

### RGB lower bound

The 80x80 RGB input alone occupies 19,200 B. During the current stem convolution, its input
and eight-channel output coexist:

```text
19,200 B input + 12,800 B output = 32,000 B
```

Consequently, a 25 KiB live-activation target is not realistic for the current 80x80 RGB
contract unless allocation reuse changes or the stem becomes extremely narrow. A practical
first RGB target is approximately 32 KiB fused live activations.

### Internal-SRAM threshold

The current actual arena use is 69,068 B, while the largest ready-state internal block is
61,440 B. Reducing arena use below that number is necessary but not automatically safe because
Wi-Fi, camera, HTTP, and runtime code still require internal memory.

- Initial target: no more than 50 KiB actual arena use.
- Safer stretch target: 40-48 KiB actual arena use.
- Keep camera framebuffers, RGB scratch, and dashboard JPEG buffers in PSRAM.
- If the target is reached, test an internal `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT` arena.
- Run camera, Wi-Fi, dashboard, and repeated inference stress tests before accepting it.

Shrinking only the configured 358,400-byte PSRAM reservation saves reserved address space but
does not reduce actual TFLM work. The important experiment is whether the complete arena can
be placed safely in internal SRAM.

### Optional luminance experiment

An 80x80 single-channel luminance model can be trained from the same COCO images; it is not a
new dataset.

| Property | RGB | Luminance |
|---|---:|---:|
| Input tensor | 19,200 B | 6,400 B |
| Stem MACs with eight output channels | 345,600 | 115,200 |

This candidate may better match the weak OV3660 color response and offers a stronger path to
lower activation memory. It may also lose useful color cues, so it remains an optional branch
and must not replace the RGB path without accuracy and device validation.

## Selected quantization techniques

### Existing quantization contract

The current FlatBuffer already uses:

- INT8 input, weights, activations, and output.
- Per-channel quantized convolution weights.
- INT32 convolution biases.
- Per-tensor activation quantization.
- ReLU6 activations.
- INT8-only TFLite built-ins with no float fallback.

Ordinary post-training INT8 conversion is therefore the reference, not a new optimization.
Further quantization work should focus on recovering accuracy after structural pruning.

### 1. INT8 quantization-aware training

For each promising pruned architecture:

1. Initialize surviving float weights from the reference network.
2. Fine-tune the float candidate.
3. Insert fake-quantization behavior matching the intended INT8 export.
4. Fine-tune at a low learning rate.
5. Export a full-INT8 FlatBuffer.
6. Confirm per-channel weight quantization and an all-INT8 operator inventory.
7. Compare float and INT8 predictions before physical-device profiling.

QAT is intended to recover F1, PR-AUC, and calibration. If the graph and tensor shapes are
unchanged, QAT alone is not expected to improve speed or memory.

### 2. Cross-layer equalization

Rescale adjacent convolution channels before final quantization so one unusually large channel
does not determine an unnecessarily broad quantization range. Account for folded BatchNorm
parameters and verify float equivalence before export.

### 3. Post-quantization bias correction

Measure systematic float-versus-INT8 output error for every convolution using existing
representative samples. Adjust biases where the correction improves validation parity without
changing the graph.

### 4. Camera-matched calibration

Calibration samples must use the same numerical preprocessing as firmware:

```text
resize and channel ordering
-> chroma reduction
-> gamma transform
-> normalization
-> input quantization
```

Compare default calibration with percentile or MSE-based clipping. Continue using the existing
dataset, with the device transform simulated exactly. Any optional device captures are for
unlabeled range calibration and parity checks, not supervised retraining.

## Techniques rejected for the first optimization cycle

| Technique | Decision | Reason |
|---|---|---|
| Unstructured magnitude pruning | Reject | Dense kernels still process zero weights; no reliable latency or activation benefit. |
| Weight clustering | Defer | May compress storage, but flash is not the bottleneck and dense compute is unchanged. |
| INT4 weights | Defer | Would require validated optimized kernels; it primarily saves non-limiting flash storage. |
| Binary or ternary weights | Reject | Requires custom kernels and a much larger model redesign. |
| Dynamic-range quantization | Reject | Does not preserve the required full-integer TFLM path. |
| Float16 quantization | Reject | Does not match the target CPU's strongest deployment path. |
| INT16 activations | Reject | Likely increases arena use and latency. |
| Arbitrary channel counts | Defer | Can introduce kernel alignment and tail-processing effects. |
| Classifier-only pruning | Reject | The classifier consumes only about 0.21 ms. |
| Arena-reservation reduction alone | Insufficient | Frees reserved PSRAM but does not reduce actual inference cost. |

## Versioned experiment sequence

| Version | Change from previous version | Primary question |
|---|---|---|
| V2.0 | Freeze current `fast_80` | Is the baseline reproducible? |
| V2.1 | Remove graph `MUL` and `ADD` | Can normalization move to firmware with equivalent output? |
| V2.2a-c | Remove each repeated block independently | Which block is least important? |
| V2.3 | Run layer-by-layer channel sensitivity | Which channels and stages tolerate narrowing? |
| V2.4 | Build the best approximately 2.8M-MAC schedule | Does structured pruning retain the accuracy gates? |
| V2.5 | Combine channel and one-block pruning near 2.5M MACs | Does it approach the 3 fps compute budget? |
| V2.6 | Apply INT8 QAT, equalization, and bias correction | How much pruned-model accuracy is recoverable? |
| V2.7 | Attempt internal-SRAM arena placement | Does lower arena use unlock a device latency step-change? |
| Optional V2.G | Train an 80x80 luminance candidate | Is color helping enough to justify RGB cost? |

Only one conceptual change should be introduced per version until its isolated effect is
understood.

## Training protocol

All candidate comparisons must use:

- The same train, validation, and held-out test manifests.
- The same seed or a documented multi-seed evaluation.
- The same camera-domain augmentation setting unless that setting is the isolated experiment.
- Identical early stopping and threshold-selection rules.
- Initialization from surviving `fast_80` weights when tensor shapes allow it.
- A short recovery stage after pruning, followed by full convergence only for candidates that
  pass preliminary gates.

Knowledge distillation from the frozen `fast_80` model is allowed because it uses the same
training images and does not introduce another dataset. It is an accuracy-recovery technique,
not a deployment operator: only the student model is exported.

## Required measurement record for every candidate

### Offline model record

- Version and parent version.
- Exact channel schedule and removed blocks.
- Parameter count, model bytes, constant bytes, and estimated MACs.
- Float and INT8 accuracy, precision, recall, specificity, F1, ROC-AUC, and PR-AUC.
- Validation-selected threshold.
- Float-versus-INT8 probability MAE, maximum error, and decision agreement.
- Operator inventory and quantization details.
- Fused live-activation peak and peak operator.

### Physical-device record

- Exact firmware, model hash, ESP-IDF version, dependency lock, and build size.
- TFLM arena reservation, actual use, utilization, and memory capability.
- Internal SRAM free and largest block before and after model initialization.
- Mapped PSRAM free before and after initialization.
- Every-operator minimum, mean, median, p95, and maximum latency.
- Full Invoke minimum, mean, median, p95, and maximum latency.
- Preprocessing and capture latency.
- End-to-end frame period and FPS.
- At least 1,000 steady-state inference iterations for final candidates.
- Camera, Wi-Fi, dashboard, allocation, and watchdog errors.

### Behavioral record

- Empty static scene.
- Person entering.
- Person remaining still.
- Person leaving.
- Lighting transitions.
- Intended near and far viewpoints.
- Raw score, threshold result, motion value, temporal votes, and final state.

## Acceptance gates

A candidate must satisfy all hard gates before it replaces `fast_80`:

| Gate | Requirement |
|---|---|
| Runtime type | Full INT8; no float fallback |
| Batch size | Exactly one |
| Operators | All supported by the deployed TFLM resolver |
| Held-out recall | At least 0.78 |
| Held-out F1 | At least 0.70 |
| F1 loss from fast 80 | Prefer no more than 0.02 absolute |
| FlatBuffer | At most 250 KiB |
| MAC target | Prefer no more than 2.5M for the 3 fps candidate |
| Fused live activations | Prefer no more than 32 KiB for RGB; stretch goal 25 KiB |
| Actual TFLM arena | No more than 50 KiB; stretch goal 40-48 KiB |
| Arena safety margin | At least 20% above measured high-water use |
| Device timing | Must improve physical Invoke and end-to-end timing |
| Stability | No camera, Wi-Fi, web, memory, or watchdog regression |

The 3 fps goal is a design objective, not an automatic acceptance criterion. Removing the
intentional 100 ms task delay and measuring the complete pipeline are required before making a
final FPS claim.

## Decision rule

Candidates belong on a Pareto frontier. A model should survive only if no other tested model
is simultaneously:

- More accurate.
- Faster on the ESP32.
- Smaller in actual arena use.
- Smaller or equal in Flash storage.

Use F1 and PR-AUC to protect overall classification quality, recall to protect person
detection, specificity to expose false wake-ups, and physical-device latency as the speed
source of truth.

## Planned notebook

Create `12_manual_pruning_and_quantization.ipynb` with these stages:

1. Load and verify the frozen `fast_80` reference.
2. Rebuild the normalization-free equivalent model.
3. Generate block-pruned candidates.
4. Calculate per-layer channel-importance rankings.
5. Run manual channel-sensitivity experiments.
6. Assemble selected channel schedules.
7. Fine-tune and optionally distill from `fast_80`.
8. Run INT8 PTQ and QAT comparisons.
9. Verify parity, quantization parameters, operators, MACs, and activation liveness.
10. Export versioned TFLite and firmware assets.
11. Import physical-device profiles.
12. Produce a Pareto report across accuracy, latency, Flash, SRAM, PSRAM, and arena use.

The notebook must not overwrite the reference checkpoint, TFLite file, device logs, or reports.

## Source artifacts

- [Main comparison](artifacts/device_profiles/MODEL_VERSION_COMPARISON.md)
- [Whole-model measurements](artifacts/device_profiles/model_version_comparison.csv)
- [Every-operator measurements](artifacts/device_profiles/operator_version_comparison.csv)
- [Fast-80 layer report](artifacts/device_profiles/fast_80/DEVICE_LAYER_PROFILE_REPORT.md)
- [Hardware capacity report](firmware/esp32_cam_vww/HARDWARE_CAPACITY_REPORT.md)
- [Firmware build and smoke-test report](firmware/esp32_cam_vww/BUILD_REPORT.md)
- [Current model definition](src/vww_esp32/modeling.py)
- [Current full-INT8 exporter](src/vww_esp32/exporting.py)

## Recommended first action

Implement and validate V2.1, the normalization-free but otherwise equivalent model. It is the
lowest-risk way to remove measured device latency and gives every subsequent pruning candidate
a cleaner baseline. After V2.1 passes numerical and device parity, begin the three independent
block-removal experiments before performing channel sensitivity analysis.
