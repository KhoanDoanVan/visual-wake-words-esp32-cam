# ESP32-CAM versus ESP32-S3 hardware comparison

Measured/inspected on 2026-09-22. This report compares the two physical development
platforms available to this project:

- the original AI-Thinker-pinout ESP32-CAM/ESP32-CAM-MB with OV3660; and
- the newly connected ESP32-S3 board with 16 MB flash and 8 MB embedded PSRAM.

The ESP32-CAM has run the complete camera and VWW pipeline. The ESP32-S3 has now run
the same frozen Fast80 FlatBuffer in an isolated physical-device profiler, including
internal/PSRAM placement, per-operator latency, memory high-water marks, and copy
bandwidth. It is not yet an end-to-end camera comparison, but the later physical camera
probe identified the S3 sensor and validated its camera pin map and resolution range.

## Executive conclusion

The ESP32-S3 is the stronger efficient-ML target. Although both chips have two CPU
cores and a maximum 240 MHz clock, the S3 replaces the LX6 cores with LX7 cores and
adds 128-bit vector instructions. ESP-NN selects S3-specific assembly kernels, while
the original ESP32 receives generic optimized kernels. The connected S3 also has four
times the flash capacity and can potentially make its full 8 MB PSRAM more useful than
the original ESP32 build, which detected 8 MB but could map only 4 MB.

The physical same-model result confirms the expected advantage: Fast80 takes 67.105 ms
with its arena in internal SRAM and 73.478 ms with its arena in PSRAM. The equivalent
instrumented internal-arena result is 6.78x faster than the saved ESP32-CAM Fast80
profile. This comparison isolates Invoke; camera input and preprocessing are not yet
equivalent across boards.

The original ESP32-CAM remains the only currently validated camera platform. Its
OV3660 sensor, pin map, LEDs, PSRAM allocation, Wi-Fi dashboard, and complete pipeline
have all been exercised. The S3 camera is now confirmed as an OV3660 and its
ESP32-S3-EYE-style pin map successfully captures JPEG frames. The exact board product
name and LED GPIO assignments remain unknown.

## Physical identity and capacity

| Property | ESP32-CAM (measured) | ESP32-S3 (physical profiler) | Interpretation |
|---|---:|---:|---|
| SoC | ESP32-D0WD-V3 rev 3.1 | ESP32-S3 QFN56 rev 0.2 | Different firmware targets and binaries |
| CPU | 2x Xtensa LX6, up to 240 MHz | 2x Xtensa LX7, up to 240 MHz | Same peak clock; newer S3 core/instructions |
| Runtime CPU clock | 240 MHz | 240 MHz measured | Equal clock; architecture/kernels drive the gain |
| Neural-network acceleration | No dedicated accelerator; generic ESP-NN optimization | 128-bit vector instructions; S3 ESP-NN assembly | Main expected inference advantage |
| Nominal on-chip SRAM | 520 KiB | 512 KiB | S3 is not an on-chip-SRAM capacity upgrade |
| Physical PSRAM detected | 8 MiB | 8 MiB embedded, `AP_3v3` | Equal detected capacity |
| PSRAM mapped/usable | 4,191,744 B before allocations | 8,388,608 B | S3 maps the complete 8 MiB |
| PSRAM runtime clock | 40 MHz | 80 MHz octal | S3 has a substantially stronger external-memory path |
| SPI flash | 4 MiB, DIO, 40 MHz | 16 MiB, DIO, 80 MHz | S3 has 4x capacity and 2x configured clock |
| Crystal | 40 MHz | 40 MHz | Equal |
| Wi-Fi | 2.4 GHz 802.11b/g/n | 2.4 GHz 802.11b/g/n | Similar radio class |
| Bluetooth | Bluetooth Classic + BLE 4.2 class | Bluetooth LE 5 class; no Classic BT | Different Bluetooth feature set |
| Secure Boot | Not recorded in the saved profile | Disabled | S3 is currently development-friendly |
| Flash encryption | Not recorded in the saved profile | Disabled | S3 flash is currently unencrypted |
| Host connection used | ESP32-CAM-MB USB-UART | QinHeng `0x1a86:0x55d3` UART bridge | Both support serial development |
| Native USB | None | USB Serial/JTAG and USB 2.0 OTG capability | S3 adds native USB/debug options |

The S3's detected 16 MB flash plus 8 MB PSRAM is consistent with an N16R8 module
configuration. That suffix must remain provisional until the module shield marking or
board documentation is inspected; esptool identifies capacities, not the complete
development-board product.

## Camera and complete-pipeline status

| Capability | ESP32-CAM | ESP32-S3 |
|---|---:|---:|
| Camera sensor | OV3660 confirmed, PID `0x3660` | OV3660 confirmed, PID `0x3660`, SCCB `0x3c` |
| Sensor maximum | 2048x1536 | 2048x1536 JPEG captured successfully |
| Active capture | 160x120 JPEG | Diagnostic sweep from 160x120 through 2048x1536 JPEG |
| Model input | 80x80x3 INT8 | Same Fast80 artifact deployed and verified |
| Camera-only throughput | 6.94 fps | 72.0 ms/frame at 320x240; 179.8 ms/frame at 2048x1536 (3-sample probe) |
| JPEG decode + preprocessing | About 60 ms | Not measured |
| Dashboard/Wi-Fi AP | Working at `192.168.4.1` | Not ported |
| Inference indicator | GPIO33 red + GPIO4 white flash | Pin assignment unknown |
| Full steady pipeline | Approximately 1.6-1.7 fps | Not measured |

The existing camera firmware must not be flashed to the S3 unchanged. It targets
`esp32`, assumes the AI-Thinker camera pin map, and assigns LEDs to GPIOs belonging to
that board. An S3 port requires the exact board schematic or pin map first.

## Fast-80 deployment comparison

The frozen model contract is the same candidate for both platforms:

| Model item | Value |
|---|---:|
| Artifact | `artifacts/fast_80/models/vww_mobilenetv1_80_int8.tflite` |
| SHA-256 | `70eb89724ae7f5f84097922e481ba8c61946ef7219fdbeddabc81a612ae7691d` |
| Input | 1x80x80x3 INT8, batch 1 |
| Input bytes | 19,200 B |
| Parameters | 111,793 |
| Dense MACs | 3,993,536 per inference |
| TFLite size | 167,976 B |
| Fused live activation peak | 38,400 B |
| Measured TFLM arena use on ESP32 | 69,068 B |

| Metric | ESP32-CAM | ESP32-S3 |
|---|---:|---:|
| Fast-80 instrumented Invoke mean | 455.546 ms | 67.164 ms internal / 73.525 ms PSRAM |
| Robust 100-sample Invoke mean | Not recorded | 67.105 ms internal / 73.478 ms PSRAM |
| Robust median / p95 | Not recorded | 67.107 / 67.119 ms internal |
| Effective Fast-80 throughput | About 8.77 MMAC/s | 59.51 MMAC/s internal / 54.35 PSRAM |
| Actual TFLM arena use | 69,068 B | 83,420 B |
| Tensor-arena placement | PSRAM | Both internal SRAM and PSRAM measured |
| ESP-NN implementation class | Generic optimized kernels | S3 vector assembly confirmed in build |

The later 416.805 ms ESP32 measurement belongs to the selected iterative-s50 pruning
capture, not the frozen Fast-80 reference. Because the sparse model retains dense
shapes and was measured in a separate session, that value must not be used as evidence
of sparse acceleration or substituted into the controlled Fast-80 hardware row.

## What the official ESP-NN benchmark does—and does not—show

Espressif's ESP-NN repository reports the following model-level INT8 Visual Wake Words
Invoke measurements with ESP-NN enabled:

| Espressif reference benchmark | ESP32 | ESP32-S3 |
|---|---:|---:|
| VWW Invoke | 380 ms | 54 ms |
| Relative result | 1.0x | About 7.0x faster |

Espressif also reports 47 ms for the S3 when the reference person-detection arena is in
internal RAM. This supports selecting the S3 for the next port, but it is **external
reference evidence**, not a prediction or measurement for Fast-80. Differences in
model topology, ESP-IDF/ESP-NN version, tensor placement, profiling hooks, camera work,
and memory traffic can materially change the result.

The project-specific S3 measurements are 67.1 ms internally and 73.5 ms in PSRAM,
slower than Espressif's 47/54 ms reference but in the same performance class. The
difference is expected because the model topology, component versions, and profiling
protocol differ. Camera bandwidth and complete-pipeline FPS remain unmeasured.

## Memory implications for this project

### Original ESP32-CAM

- The 350 KiB arena reservation cannot fit in the largest internal allocation.
- The arena therefore lives in mapped PSRAM.
- Internal SRAM copy throughput measured 181.55 MiB/s.
- Mapped 40 MHz PSRAM copy throughput measured only 4.71 MiB/s.
- Ready-state internal free heap was approximately 66.8 KiB.
- Memory traffic is therefore a major part of inference cost.

### Connected ESP32-S3

- Runtime allocatable internal 8-bit heap is 466,651 B, with a 253,952-byte largest
  block at probe start.
- All 8,388,608 B of octal PSRAM are mapped; sustained 256 KiB PSRAM copy measured
  21.30 MiB/s versus 4.71 MiB/s on the older board.
- A 163,840-byte internal arena succeeds and uses 83,420 B, leaving a lifetime minimum
  of 219,003 B internal heap in the isolated probe.
- Internal placement reduces Invoke by 8.67% versus PSRAM, while preserving identical
  output on the fixed input.
- Camera and Wi-Fi initialization may fragment or consume enough SRAM to invalidate
  the isolated internal-arena allocation, so the allocation must be repeated in the
  final application before it is adopted.

## Required software-port changes

The model target, memory configuration, operator contract, numerical parity, and
ESP-NN kernel selection have now been validated by the capacity probe. The remaining
camera-application port requires:

1. Record the exact board product name from its PCB marking; the OV3660 sensor is confirmed.
2. Use the validated ESP32-S3-EYE-style camera GPIO map from `esp32_s3_camera_probe`.
3. Assign safe LED/flash GPIOs instead of carrying GPIO33/GPIO4 assumptions forward.
4. Revalidate JPEG capture, RGB channel order, resize geometry, exposure, and color
   preprocessing on the S3 camera.
5. Keep the flash disabled during controlled accuracy and latency calibration because
   illumination changes the input distribution.

## Controlled same-model benchmark protocol

The model-only phase of this protocol is complete. It used the exact Fast80 hash,
batch one, identical 80x80x3 INT8 input, two warm-ups plus 20 per-operator samples,
100 robust total-latency samples, both arena placements, and output-parity checks. Its
results are in `artifacts/device_profiles/esp32_s3_capacity/DEVICE_CAPACITY_REPORT.md`.

The remaining controlled phase is to benchmark the complete camera pipeline with
equivalent capture, preprocessing, dashboard, motion gate, and frame interval. Flash
illumination must remain disabled during this calibration. The final run must repeat
heap/arena measurements after camera and Wi-Fi initialization because those subsystems
can change whether the internal arena fits.

The resulting report should include:

- mean/min/max and preferably median/p95 Invoke latency;
- camera-only FPS and complete-pipeline FPS;
- actual TFLM arena usage and reservation;
- fused live-activation peak;
- internal/PSRAM free and largest blocks;
- internal/PSRAM copy bandwidth;
- flash image and model sizes; and
- numerical parity on a fixed labeled input set.

## Decision

Use the ESP32-S3 as the next performance-port target, while retaining the ESP32-CAM as
the validated camera-functional baseline. The same Fast80 artifact has now demonstrated
a 6.78x model-only speedup with an internal arena. Do not present this as a full-camera
speedup until the S3 camera and preprocessing pipeline are ported and measured.

The S3's vector-aware ESP-NN path may change the optimization priority: a model that is
too slow on the original ESP32 may already meet the frame-rate target on S3. Structured
channel pruning and activation reduction remain valuable for latency, memory traffic,
and energy, but their acceptance gates should be re-measured on both platforms rather
than inherited from the original ESP32 profile.

## Evidence and sources

### Physical project evidence

- ESP32-CAM measurements: `firmware/esp32_cam_vww/HARDWARE_CAPACITY_REPORT.md`
- Fast-80 layer profile: `artifacts/device_profiles/fast_80/DEVICE_LAYER_PROFILE_REPORT.md`
- Fast-80 raw log: `artifacts/device_profiles/fast_80/device_serial.log`
- ESP32-S3 physical profiler:
  `artifacts/device_profiles/esp32_s3_capacity/DEVICE_CAPACITY_REPORT.md`
- ESP32-S3 structured data:
  `artifacts/device_profiles/esp32_s3_capacity/device_profile_summary.json` and
  `operator_profile.csv`
- ESP32-S3 ROM inspection: `esptool chip_id`, `flash_id`, and `get_security_info` on
  the connected UART port. The device MAC and transient serial path are intentionally
  omitted from this repository report.

### Primary documentation

- [ESP32 series datasheet](https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf)
- [ESP32-S3 series datasheet](https://documentation.espressif.com/esp32_s3_datasheet_en.pdf)
- [ESP-NN optimized-kernel repository and reference benchmarks](https://github.com/espressif/esp-nn)
- [ESP32-S3 USB Serial/JTAG documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/usb-serial-jtag-console.html)
- [Espressif module part-number guide](https://developer.espressif.com/blog/2025/03/espressif-part-numbers-explained/)
