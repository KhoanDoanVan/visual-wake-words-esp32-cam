# ESP32-CAM VWW model comparison: 96x96 vs 80x80

Both full-INT8 models were measured on the same ESP32-CAM with batch size one.
Device latency is the mean of 20 invocations after two warm-up invocations.
Accuracy metrics use the same 2,000-image held-out COCO-derived test set, but each
model uses its own validation-selected threshold.

> Both latency runs use the same lightweight per-operator profiler. This makes the
> variant comparison controlled, but the absolute values include instrumentation cost.
> Use the uninstrumented production firmware for final FPS and response-time claims.

## Whole-model comparison

| Perspective | Baseline 96 | Fast 80 | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Parameters | 111,793 | 111,793 | +0 | +0.0% |
| TFLite storage (B) | 167,976 | 167,976 | +0 | +0.0% |
| MACs / inference | 5,616,256 | 3,993,536 | -1,622,720 | -28.9% |
| Keras graph peak estimate (B) | 73,728 | 51,200 | -22,528 | -30.6% |
| TFLite graph live peak (B) | 55,296 | 38,400 | -16,896 | -30.6% |
| TFLM arena used (B) | 91,596 | 69,068 | -22,528 | -24.6% |
| TFLM arena utilization | 25.6% | 19.3% | -6.3% | -24.6% |
| Internal SRAM free at ready (B) | 66,867 | 66,867 | +0 | +0.0% |
| Largest internal SRAM block (B) | 61,440 | 61,440 | +0 | +0.0% |
| Mapped PSRAM free at ready (B) | 3,378,736 | 3,387,184 | +8,448 | +0.3% |
| PSRAM consumed to ready (B) | 813,008 | 804,560 | -8,448 | -1.0% |
| Instrumented Invoke mean (ms) | 640.534 | 455.546 | -184.988 | -28.9% |
| Effective throughput (MMAC/s) | 8.768 | 8.766 | -0.002 | -0.0% |
| Accuracy | 0.6910 | 0.6575 | -0.0335 | -4.8% |
| Precision | 0.6449 | 0.6010 | -0.0439 | -6.8% |
| Recall | 0.8236 | 0.8981 | +0.0744 | +9.0% |
| Specificity | 0.5633 | 0.4259 | -0.1374 | -24.4% |
| F1 | 0.7234 | 0.7201 | -0.0033 | -0.5% |
| PR-AUC | 0.8034 | 0.7873 | -0.0162 | -2.0% |
| Decision threshold | 0.34 | 0.27 | -0.07 | — |

Measured Invoke ranges were 638.395-644.099 ms
for baseline 96 and 453.364-459.697 ms
for fast 80.

The 80x80 model changes spatial resolution but not channel widths, so parameter and
FlatBuffer sizes remain essentially constant. Its value is lower activation traffic and
fewer MACs. The latency result is a physical-device measurement, while peak live INT8
activation is a graph-liveness estimate and arena usage is reported by TFLite Micro.

## Memory interpretation

- The firmware reserves 358,400 B of mapped PSRAM for
  the tensor arena in both versions. TFLite Micro actually uses
  91,596 B for baseline 96 and
  69,068 B for fast 80.
- Exact TFLite tensor lifetimes give theoretical live-activation peaks of
  55,296 B and
  38,400 B. These sit inside arena usage;
  they are not additional allocations.
- The remaining used arena covers persistent tensors, scratch buffers, allocator metadata,
  alignment, and planner decisions. Per-operator heap sampling would not reveal these because
  TFLite Micro allocates the arena before inference and reuses it without heap allocation.
- Internal SRAM free at ready remains 66,867 B with a
  largest contiguous block of 61,440 B, so the
  tensor arena must stay in PSRAM.
- Opening the MJPEG stream conditionally allocates another
  184,320 B
  client buffer. It is not included in the ready-state heap snapshot.

## Operators contributing the most saved latency

| Index | Operator | Baseline ms | Fast-80 ms | Saved ms | Reduction |
|---:|---|---:|---:|---:|---:|
| 6 | CONV_2D | 79.710 | 54.282 | 25.428 | 31.9% |
| 2 | CONV_2D | 69.301 | 47.748 | 21.553 | 31.1% |
| 8 | CONV_2D | 46.937 | 31.589 | 15.348 | 32.7% |
| 4 | CONV_2D | 49.573 | 34.592 | 14.981 | 30.2% |
| 1 | ADD | 37.718 | 24.752 | 12.967 | 34.4% |
| 12 | CONV_2D | 40.366 | 27.989 | 12.376 | 30.7% |
| 18 | CONV_2D | 38.174 | 27.154 | 11.020 | 28.9% |
| 20 | CONV_2D | 37.336 | 26.940 | 10.395 | 27.8% |

## Accuracy trade-off

- F1 changes from 0.7234 to 0.7201
  (-0.0033).
- Recall changes from 0.8236 to 0.8981
  (+0.0744); specificity changes by -0.1374.
- These COCO-domain metrics do not replace evaluation on labeled OV3660 frames.

## Optimization decision

Use the operator comparison to reduce channels in layers with both high measured latency
and high activation traffic. The next 3 fps candidate should target no more than about
2.4M MACs, at most 25 KiB theoretical TFLite live activations, and at most 50 KiB actual
TFLM arena usage. It should then be fine-tuned and calibrated on real OV3660 captures.
