# ESP32-CAM device operator profile: student_120_hard_control

This report contains timings measured inside TFLite Micro on the physical ESP32-CAM.
It uses batch 1, 20 measured invocations after 2 warm-up
invocations, and input `120x120x3`.

> These timings include lightweight per-operator profiler hooks. Use them for operator
> attribution and controlled comparisons between identically instrumented builds. The
> uninstrumented production firmware remains the source of truth for final user-facing
> latency and FPS.

## Summary

| Metric | Value |
|---|---:|
| Full Invoke mean | 1117.123 ms |
| Full Invoke range | 1114.846-1121.736 ms |
| Sum of profiled operators | 1115.925 ms |
| Dispatch/profiler remainder | 1.198 ms |
| Tensor arena actually used | 142,444 B |
| Tensor arena reservation | 358,400 B |
| Tensor arena utilization | 39.7% |
| TFLite live activation peak | 86,400 B at operator 0 |
| Arena bytes beyond live-tensor estimate | 56,044 B |
| Internal SRAM free / largest block at ready | 66,799 / 61,440 B |
| Mapped PSRAM free at ready | 3,178,860 B |
| TFLite model | 304,296 B |
| Operator constants in FlatBuffer | 218,878 B |
| Estimated MACs | 10,487,648 |
| Fused TFLite operators | 34 |

## Ten slowest operators

| Index | Operator | Input | Output | MACs | Mean ms | Latency share |
|---:|---|---|---|---:|---:|---:|
| 30 | CONV_2D | 1x3x3x256 | 1x3x3x256 | 589,824 | 114.015 | 10.2% |
| 2 | CONV_2D | 1x120x120x3 | 1x60x60x8 | 777,600 | 108.406 | 9.7% |
| 8 | CONV_2D | 1x30x30x32 | 1x30x30x32 | 921,600 | 73.717 | 6.6% |
| 4 | CONV_2D | 1x60x60x8 | 1x60x60x16 | 460,800 | 72.028 | 6.5% |
| 12 | CONV_2D | 1x15x15x64 | 1x15x15x64 | 921,600 | 69.011 | 6.2% |
| 1 | ADD | 1x120x120x3 | 1x120x120x3 | 0 | 60.422 | 5.4% |
| 17 | CONV_2D | 1x7x7x128 | 1x7x7x128 | 802,816 | 51.555 | 4.6% |
| 19 | CONV_2D | 1x7x7x128 | 1x7x7x128 | 802,816 | 51.309 | 4.6% |
| 25 | CONV_2D | 1x7x7x128 | 1x7x7x128 | 802,816 | 51.017 | 4.6% |
| 23 | CONV_2D | 1x7x7x128 | 1x7x7x128 | 802,816 | 50.706 | 4.5% |

## Operators with the largest activation footprint

| Index | Operator | Input | Output | Activation I/O | Live during op | Constants |
|---:|---|---|---|---:|---:|---:|
| 0 | MUL | 1x120x120x3 | 1x120x120x3 | 86,400 | 86,400 | 1 |
| 1 | ADD | 1x120x120x3 | 1x120x120x3 | 86,400 | 86,400 | 1 |
| 4 | CONV_2D | 1x60x60x8 | 1x60x60x16 | 86,400 | 86,400 | 192 |
| 2 | CONV_2D | 1x120x120x3 | 1x60x60x8 | 72,000 | 72,000 | 248 |
| 5 | DEPTHWISE_CONV_2D | 1x60x60x16 | 1x30x30x16 | 72,000 | 72,000 | 208 |
| 3 | DEPTHWISE_CONV_2D | 1x60x60x8 | 1x60x60x8 | 57,600 | 57,600 | 104 |
| 7 | DEPTHWISE_CONV_2D | 1x30x30x32 | 1x30x30x32 | 57,600 | 57,600 | 416 |
| 8 | CONV_2D | 1x30x30x32 | 1x30x30x32 | 57,600 | 57,600 | 1,152 |
| 6 | CONV_2D | 1x30x30x16 | 1x30x30x32 | 43,200 | 43,200 | 640 |
| 9 | DEPTHWISE_CONV_2D | 1x30x30x32 | 1x15x15x32 | 36,000 | 36,000 | 416 |

## Device memory accounting

`memory_summary.csv` separates measured heap values, configured allocations, and static
FlatBuffer estimates. Important nesting rules apply: arena-used bytes are inside the arena
reservation, and live activation bytes are inside arena-used bytes. Model constants remain
in flash and must not be added to SRAM or PSRAM use. The stream-client buffer is allocated
only while a browser consumes the MJPEG endpoint.

The theoretical live-activation peak is 86,400 B. TFLite Micro reports
142,444 B used in its arena, leaving 215,956 B
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
