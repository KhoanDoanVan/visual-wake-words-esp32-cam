# ESP32-CAM device operator profile: baseline_96

This report contains timings measured inside TFLite Micro on the physical ESP32-CAM.
It uses batch 1, 20 measured invocations after 2 warm-up
invocations, and input `96x96x3`.

> These timings include lightweight per-operator profiler hooks. Use them for operator
> attribution and controlled comparisons between identically instrumented builds. The
> uninstrumented production firmware remains the source of truth for final user-facing
> latency and FPS.

## Summary

| Metric | Value |
|---|---:|
| Full Invoke mean | 640.534 ms |
| Full Invoke range | 638.395-644.099 ms |
| Sum of profiled operators | 639.423 ms |
| Dispatch/profiler remainder | 1.111 ms |
| Tensor arena actually used | 91,596 B |
| Tensor arena reservation | 358,400 B |
| Tensor arena utilization | 25.6% |
| TFLite live activation peak | 55,296 B at operator 0 |
| Arena bytes beyond live-tensor estimate | 36,300 B |
| Internal SRAM free / largest block at ready | 66,867 / 61,440 B |
| Mapped PSRAM free at ready | 3,378,736 B |
| TFLite model | 167,976 B |
| Operator constants in FlatBuffer | 111,806 B |
| Estimated MACs | 5,616,256 |
| Fused TFLite operators | 26 |

## Ten slowest operators

| Index | Operator | Input | Output | MACs | Mean ms | Latency share |
|---:|---|---|---|---:|---:|---:|
| 6 | CONV_2D | 1x24x24x16 | 1x24x24x32 | 294,912 | 79.710 | 12.5% |
| 2 | CONV_2D | 1x96x96x3 | 1x48x48x8 | 497,664 | 69.301 | 10.8% |
| 4 | CONV_2D | 1x48x48x8 | 1x48x48x16 | 294,912 | 49.573 | 7.8% |
| 8 | CONV_2D | 1x24x24x32 | 1x24x24x32 | 589,824 | 46.937 | 7.3% |
| 12 | CONV_2D | 1x12x12x64 | 1x12x12x64 | 589,824 | 40.366 | 6.3% |
| 18 | CONV_2D | 1x6x6x128 | 1x6x6x128 | 589,824 | 38.174 | 6.0% |
| 1 | ADD | 1x96x96x3 | 1x96x96x3 | 0 | 37.718 | 5.9% |
| 16 | CONV_2D | 1x6x6x128 | 1x6x6x128 | 589,824 | 37.430 | 5.9% |
| 20 | CONV_2D | 1x6x6x128 | 1x6x6x128 | 589,824 | 37.336 | 5.8% |
| 22 | CONV_2D | 1x3x3x128 | 1x3x3x256 | 294,912 | 34.461 | 5.4% |

## Operators with the largest activation footprint

| Index | Operator | Input | Output | Activation I/O | Live during op | Constants |
|---:|---|---|---|---:|---:|---:|
| 0 | MUL | 1x96x96x3 | 1x96x96x3 | 55,296 | 55,296 | 1 |
| 1 | ADD | 1x96x96x3 | 1x96x96x3 | 55,296 | 55,296 | 1 |
| 4 | CONV_2D | 1x48x48x8 | 1x48x48x16 | 55,296 | 55,296 | 192 |
| 2 | CONV_2D | 1x96x96x3 | 1x48x48x8 | 46,080 | 46,080 | 248 |
| 5 | DEPTHWISE_CONV_2D | 1x48x48x16 | 1x24x24x16 | 46,080 | 46,080 | 208 |
| 3 | DEPTHWISE_CONV_2D | 1x48x48x8 | 1x48x48x8 | 36,864 | 36,864 | 104 |
| 7 | DEPTHWISE_CONV_2D | 1x24x24x32 | 1x24x24x32 | 36,864 | 36,864 | 416 |
| 8 | CONV_2D | 1x24x24x32 | 1x24x24x32 | 36,864 | 36,864 | 1,152 |
| 6 | CONV_2D | 1x24x24x16 | 1x24x24x32 | 27,648 | 27,648 | 640 |
| 9 | DEPTHWISE_CONV_2D | 1x24x24x32 | 1x12x12x32 | 23,040 | 23,040 | 416 |

## Device memory accounting

`memory_summary.csv` separates measured heap values, configured allocations, and static
FlatBuffer estimates. Important nesting rules apply: arena-used bytes are inside the arena
reservation, and live activation bytes are inside arena-used bytes. Model constants remain
in flash and must not be added to SRAM or PSRAM use. The stream-client buffer is allocated
only while a browser consumes the MJPEG endpoint.

The theoretical live-activation peak is 55,296 B. TFLite Micro reports
91,596 B used in its arena, leaving 266,804 B
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
