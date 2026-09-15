# ESP32-CAM hardware capacity report

Measured on 2026-09-15 using the connected ESP32-CAM-MB, ESP32-CAM board,
OV3660 camera, ESP-IDF 5.3, and the 80x80 full-INT8 VWW firmware. These are
batch-size-1 measurements from the physical board, not desktop estimates.

## Executive conclusion

The camera and network are not the current throughput bottlenecks. TFLite
Micro inference is about 84% of the active pipeline time. The tensor arena is in
40 MHz mapped PSRAM, whose measured sequential copy throughput is 4.71 MiB/s,
versus 181.55 MiB/s for internal SRAM. The best next model optimization is
therefore to reduce both MACs and live activation traffic while keeping batch
size fixed at one.

The present system sustains about 1.67 fps. A realistic 3 fps target requires an
inference time near 256 ms after removing the intentional 100 ms delay, which
corresponds approximately to a 2.4M-MAC model on this runtime. A 5 fps target
requires about 123 ms inference and roughly 1.2M MACs. These are
planning estimates because latency is not perfectly linear with MAC count and
is strongly affected by PSRAM traffic.

## Processor and memory

| Capability | Hardware/specification | Observed in this firmware |
|---|---:|---:|
| SoC | ESP32-D0WD-V3 revision 3.1 | Confirmed by ROM/esptool |
| CPU | 2x Xtensa LX6, up to 240 MHz | 240 MHz |
| ML accelerator | None | TFLite Micro/ESP-NN on CPU; one image per invocation |
| On-chip SRAM | 520 KiB total across ESP32 memory regions | 261,679 B heap-free before application allocations; 66,811 B free when ready |
| Largest internal allocation | Memory-map dependent | 110,592 B before allocations; 61,440 B when ready |
| Linked DRAM | 180,736 B linkable in this build | 49,160 B used (27.20%) |
| Linked IRAM | 131,072 B linkable in this build | 111,238 B used (84.87%); 19,834 B remains |
| Physical PSRAM found | 8 MiB | Boot memory test passed |
| Directly mapped PSRAM | ESP32 address-space maximum 4 MiB | 4,191,744 B free before allocations; 3,202,860 B free when ready |
| PSRAM clock | Build requests 80 MHz | Runtime reports 40 MHz |
| Internal SRAM memcpy | N/A | 181.55 MiB/s, 8 MiB copied |
| Mapped PSRAM memcpy | N/A | 4.71 MiB/s, 8 MiB copied |

The 350 KiB TFLite tensor arena cannot fit in the largest available internal
SRAM block. It is therefore allocated in PSRAM. The 51,200-byte model
activation estimate is not the complete TFLite Micro arena requirement: the
arena also contains persistent tensors, allocator metadata, operator scratch
buffers, alignment loss, and other runtime state.

## Flash storage

| Item | Capacity/usage |
|---|---:|
| Physical SPI flash | 4 MiB, DIO, 40 MHz |
| JEDEC identification | Manufacturer `0x5e`, device `0x4016`, 3.3 V |
| NVS partition | 24 KiB |
| PHY initialization partition | 4 KiB |
| Factory application partition | 3 MiB |
| Current application binary | 1,250,960 B (39.77% of application partition) |
| Free application-partition space | 1,894,768 B |
| Embedded INT8 model | 167,976 B |

Flash capacity is not currently restrictive. Increasing model weights would
fit in storage, but would usually worsen inference latency and tensor-arena
pressure. The weights are compiled as flash-resident read-only data rather
than copied into a 168 KiB internal-SRAM allocation.

## Camera capacity and active configuration

| Capability | OV3660/driver maximum | Active ML pipeline |
|---|---:|---:|
| Sensor | OV3660, 1/5-inch color | PID `0x3660`, SCCB address `0x3c` |
| Maximum sensor output | 2048x1536 (3 MP) | 160x120 (QQVGA) |
| Supported formats | Raw RGB, RGB565/555/444, YCbCr, JPEG/compressed | JPEG capture, decoded to RGB888 |
| Camera bus | 8-bit parallel DVP | 10 MHz XCLK; sensor reports 5 MHz PCLK |
| Frame buffers | Driver/configuration dependent | One 184,320-byte-capacity buffer in PSRAM |
| JPEG quality setting | 0-63 driver scale | 12 |
| Typical current JPEG | Scene dependent | 1,830-1,850 B in the final smoke-test scene |
| Model input | N/A | RGB, 80x80x3 INT8 = 19,200 B |

The sensor's 3 MP maximum is suitable for still capture, not this always-on
inference path. On the original ESP32, Espressif warns that raw RGB/YUV capture
puts heavy strain on memory and can lose data, especially with Wi-Fi enabled.
This firmware therefore uses the sensor's JPEG output and decodes it before
inference. Direct RGB565 capture was also unstable on this physical board.

## Camera-only and batch-1 pipeline benchmarks

With inference disabled for the test, 20 consecutive captures took 2,881.4 ms:
**6.94 camera frames/s**. Their JPEGs averaged 1,838.6 B (range 1,830-1,850 B),
or 102.1 kbit/s of image payload. This isolates the sensor and driver from ML.

Five steady-state full-pipeline samples were measured. The initial diagnostic frame was
excluded because it prints a large ASCII preview and waits for the first live
camera frame.

| Stage | Mean | Stable observed range | Share of active time |
|---|---:|---:|---:|
| Acquire ready camera frame | 0.4 ms | 0.4 ms | 0.1% |
| JPEG decode + RGB correction + resize + camera-domain transform + INT8 quantization | about 60.3 ms | 59.8-61.3 ms | 12.2% |
| TFLite Micro inference, batch 1 | 416.85 ms | 414.5-422.4 ms | 84.0% |
| Copy JPEG to dashboard buffer | 0.4 ms | 0.4 ms | 0.1% |
| Statistics, motion gate, state, logging and other work | about 19 ms | Derived | 3.8% |
| Total active pipeline | about 496 ms | 494.9-497.7 ms | 100% |
| Intentional task delay | 100 ms | Fixed | Outside active time |
| Frame period | about 599 ms | 597-601 ms | 1.66-1.68 fps |

Camera acquisition appears short in steady state because the single camera
buffer is returned before inference, allowing the sensor/driver to acquire the
next JPEG while the CPU runs the model. The first non-overlapped live capture
took about 143 ms. The camera-only benchmark confirms that the sensor/driver
can supply frames about four times faster than the complete pipeline. The
current transform handles a 57,600-byte RGB888 scratch frame at an effective
end-to-end rate of roughly 0.95 MiB/s.

## Communications bandwidth

| Interface | Theoretical/configured | Observed workload |
|---|---:|---:|
| Wi-Fi | 2.4 GHz 802.11b/g/n; ESP32 documentation reports up to 20 Mbit/s TCP over the air | Current JPEG payload is about 3,070 B/s (24.6 kbit/s) before HTTP overhead |
| Camera DVP | 8 data bits; sensor reports 5 MHz PCLK | Nominal active-byte clock ceiling 5 MB/s; JPEG payload is much smaller |
| USB-UART flashing | 460,800 baud configured | esptool reported 483.4 kbit/s effective compressed flashing throughput |
| Runtime serial log | 115,200 baud | About 11.5 kB/s maximum application payload with 8-N-1 framing |

The Wi-Fi figure is an Espressif laboratory maximum, not a measurement of this
antenna, room, AP mode, or browser. Even a very large real-world reduction
would still leave ample bandwidth for the present 24.6 kbit/s image payload.
No client-side iperf test was run because the development Mac was not connected
to the isolated `VWW-Camera` access point during profiling.

## Current model footprint

| Model property | Value |
|---|---:|
| Input/batch | 1x80x80x3 INT8 |
| Output | 1x1 INT8 person probability |
| Parameters | 111,793 |
| Estimated MACs | 3,993,536 per image |
| TFLite file | 167,976 B |
| Keras-graph peak live INT8 estimate | 51,200 B |
| Fused-TFLite live activation peak | 38,400 B |
| TFLite Micro arena actually used | 69,068 B |
| Tensor arena reservation | 358,400 B in PSRAM |
| Measured throughput | About 9.58 million model MACs/s |

The fused-TFLite peak comes from tensor producer/consumer lifetimes in the exact
deployed FlatBuffer. The larger Keras estimate is retained for continuity with
the training profiler. Both are nested inside the measured TFLite Micro arena
usage rather than additional allocations. The difference between 69,068 B of
arena usage and 38,400 B of live tensors covers persistent tensors, scratch,
allocator metadata, alignment, and planner reuse.

The FlatBuffer contains only `ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`,
`FULLY_CONNECTED`, `LOGISTIC`, `MEAN`, and `MUL`. Batch size is fixed at one;
batching camera frames would increase latency and activation memory without
helping this real-time single-camera use case.

## Model-design budget for the next iteration

| Goal | Approximate inference budget | Approximate model budget | Other required change |
|---|---:|---:|---|
| Keep current behavior | 417 ms | 4.0M MACs | None |
| 2 fps | <=420 ms | <=4.0M MACs | Remove/reduce the 100 ms delay |
| 3 fps | <=256 ms | about 2.4M MACs | Remove the delay; reduce activation traffic |
| 5 fps | <=123 ms | about 1.2M MACs | Much smaller network and likely tighter preprocessing/runtime work |

Recommended hard deployment limits for the next training search:

- Full INT8 input, weights, activations, and output.
- Batch size exactly one.
- Model FlatBuffer at most 250 KiB; storage permits more, but latency does not.
- Prefer no more than 2.5M MACs for a 3 fps target.
- Prefer fused-TFLite live activations at or below 25 KiB, actual TFLM arena
  usage at or below 50 KiB, and branch-free execution to reduce slow PSRAM
  traffic.
- Keep 80x80 initially; optimize channel counts and block schedule before
  reducing resolution further, because the tested 64x64 candidate lost too
  much accuracy.
- Keep the current model fixed and validate the fixed-point camera transform,
  0.44 threshold, and 2.0 motion gate across intended viewpoints and lighting.

## Sources and measurement notes

- Espressif ESP32 datasheet: <https://documentation.espressif.com/esp32_datasheet_en.pdf>
- Espressif ESP32 camera driver: <https://github.com/espressif/esp32-camera/blob/master/README.md>
- Espressif ESP32 Wi-Fi performance: <https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/overview.html>
- Espressif PSRAM mapping FAQ: <https://docs.espressif.com/projects/esp-faq/en/latest/software-framework/storage/psram.html>
- Model profile: `artifacts/fast_80/reports/model_profile_summary.json`
- Runtime instrumentation: `firmware/esp32_cam_vww/main/app_main.cc`
- Per-operator memory profile: `artifacts/device_profiles/fast_80/operator_profile.csv`
- Memory comparison: `artifacts/device_profiles/memory_version_comparison.csv`

Memory-copy results measure `memcpy` payload throughput for repeated 32 KiB
internal-SRAM and 64 KiB PSRAM blocks. They are not raw bus signaling rates.
Pipeline timing uses `esp_timer_get_time()` around each production stage.
