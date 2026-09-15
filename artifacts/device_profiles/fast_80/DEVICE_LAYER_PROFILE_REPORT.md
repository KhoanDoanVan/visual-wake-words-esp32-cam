# ESP32-CAM device operator profile: fast_80

This report contains timings measured inside TFLite Micro on the physical ESP32-CAM.
It uses batch 1, 20 measured invocations after 2 warm-up
invocations, and input `80x80x3`.

> These timings include lightweight per-operator profiler hooks. Use them for operator
> attribution and controlled comparisons between identically instrumented builds. The
> uninstrumented production firmware remains the source of truth for final user-facing
> latency and FPS.

## Summary

| Metric | Value |
|---|---:|
| Full Invoke mean | 455.546 ms |
| Full Invoke range | 453.364-459.697 ms |
| Sum of profiled operators | 454.558 ms |
| Dispatch/profiler remainder | 0.989 ms |
| Tensor arena actually used | 69,068 B |
| Tensor arena reservation | 358,400 B |
| Tensor arena utilization | 19.3% |
| TFLite live activation peak | 38,400 B at operator 0 |
| Arena bytes beyond live-tensor estimate | 30,668 B |
| Internal SRAM free / largest block at ready | 66,867 / 61,440 B |
| Mapped PSRAM free at ready | 3,387,184 B |
| TFLite model | 167,976 B |
| Operator constants in FlatBuffer | 111,806 B |
| Estimated MACs | 3,993,536 |
| Fused TFLite operators | 26 |

## Ten slowest operators

| Index | Operator | Input | Output | MACs | Mean ms | Latency share |
|---:|---|---|---|---:|---:|---:|
| 6 | CONV_2D | 1x20x20x16 | 1x20x20x32 | 204,800 | 54.282 | 11.9% |
| 2 | CONV_2D | 1x80x80x3 | 1x40x40x8 | 345,600 | 47.748 | 10.5% |
| 4 | CONV_2D | 1x40x40x8 | 1x40x40x16 | 204,800 | 34.592 | 7.6% |
| 22 | CONV_2D | 1x3x3x128 | 1x3x3x256 | 294,912 | 34.490 | 7.6% |
| 8 | CONV_2D | 1x20x20x32 | 1x20x20x32 | 409,600 | 31.589 | 6.9% |
| 12 | CONV_2D | 1x10x10x64 | 1x10x10x64 | 409,600 | 27.989 | 6.2% |
| 16 | CONV_2D | 1x5x5x128 | 1x5x5x128 | 409,600 | 27.314 | 6.0% |
| 18 | CONV_2D | 1x5x5x128 | 1x5x5x128 | 409,600 | 27.154 | 6.0% |
| 20 | CONV_2D | 1x5x5x128 | 1x5x5x128 | 409,600 | 26.940 | 5.9% |
| 1 | ADD | 1x80x80x3 | 1x80x80x3 | 0 | 24.752 | 5.4% |

## Operators with the largest activation footprint

| Index | Operator | Input | Output | Activation I/O | Live during op | Constants |
|---:|---|---|---|---:|---:|---:|
| 0 | MUL | 1x80x80x3 | 1x80x80x3 | 38,400 | 38,400 | 1 |
| 1 | ADD | 1x80x80x3 | 1x80x80x3 | 38,400 | 38,400 | 1 |
| 4 | CONV_2D | 1x40x40x8 | 1x40x40x16 | 38,400 | 38,400 | 192 |
| 2 | CONV_2D | 1x80x80x3 | 1x40x40x8 | 32,000 | 32,000 | 248 |
| 5 | DEPTHWISE_CONV_2D | 1x40x40x16 | 1x20x20x16 | 32,000 | 32,000 | 208 |
| 3 | DEPTHWISE_CONV_2D | 1x40x40x8 | 1x40x40x8 | 25,600 | 25,600 | 104 |
| 7 | DEPTHWISE_CONV_2D | 1x20x20x32 | 1x20x20x32 | 25,600 | 25,600 | 416 |
| 8 | CONV_2D | 1x20x20x32 | 1x20x20x32 | 25,600 | 25,600 | 1,152 |
| 6 | CONV_2D | 1x20x20x16 | 1x20x20x32 | 19,200 | 19,200 | 640 |
| 9 | DEPTHWISE_CONV_2D | 1x20x20x32 | 1x10x10x32 | 16,000 | 16,000 | 416 |

## Device memory accounting

`memory_summary.csv` separates measured heap values, configured allocations, and static
FlatBuffer estimates. Important nesting rules apply: arena-used bytes are inside the arena
reservation, and live activation bytes are inside arena-used bytes. Model constants remain
in flash and must not be added to SRAM or PSRAM use. The stream-client buffer is allocated
only while a browser consumes the MJPEG endpoint.

The theoretical live-activation peak is 38,400 B. TFLite Micro reports
69,068 B used in its arena, leaving 289,332 B
unused inside the conservative 358,400 B reservation.
The difference between arena-used and live-tensor estimates includes persistent tensors,
operator scratch buffers, allocator metadata, alignment, and reuse decisions that cannot be
assigned reliably to a single operator from runtime timing hooks.

## Interpretation

Operator indexes refer to the deployed TFLite FlatBuffer, not unfused Keras layer indexes.
Batch normalization and ReLU are folded into quantized convolution operators during export.
`operator_profile.csv` contains every operator, tensor shape, activation I/O, live-tensor
memory before/during/after execution, constant bytes, MAC estimate, minimum/mean/maximum
device latency, latency share, and effective compute and activation throughput.
