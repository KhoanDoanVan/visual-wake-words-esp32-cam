# ESP32-S3 capacity and Fast80 physical-device report

Measured on the connected ESP32-S3 on 2026-09-22. This report is based on a
purpose-built ESP-IDF probe running on the physical board, not datasheet estimates.
The original firmware was backed up before deployment.

## Result at a glance

The connected board is a dual-core ESP32-S3 rev 0.2 running at 240 MHz with 16 MiB
flash and the complete 8 MiB octal PSRAM mapped at 80 MHz. Fast80 runs successfully
with ESP-NN's ESP32-S3 kernels. With a batch size of one, the same deterministic input
and a correctly sized 160 KiB arena reservation:

- internal-SRAM arena: **67.105 ms mean**, 67.107 ms median, 67.119 ms p95;
- PSRAM arena: **73.478 ms mean**, 73.477 ms median, 73.494 ms p95;
- internal placement is **8.67% lower latency** (1.095x faster) than PSRAM;
- both placements return the identical raw INT8 output `15`, or `0.55859375` after
  dequantization; and
- actual TFLM arena use is **83,420 bytes** in both placements.

Compared using the same 20-sample instrumented profiler, the internal-SRAM S3 result
is 6.78x faster than the 455.546 ms ESP32-CAM Fast80 baseline. The PSRAM-arena S3
result is 6.20x faster. These figures isolate model inference; they do not include
camera capture, JPEG decoding, resizing, Wi-Fi, or the web dashboard.

## Physical configuration

| Property | Measured value |
|---|---:|
| SoC | ESP32-S3 QFN56 rev 0.2 |
| CPU | 2 cores, 240,000,000 Hz |
| ESP-IDF | v5.3 |
| esp-tflite-micro | 1.4.1 |
| ESP-NN | 1.4.0, S3-specific assembly compiled |
| Flash | 16,777,216 B (16 MiB), DIO, 80 MHz |
| PSRAM | 8,388,608 B (8 MiB), octal, 80 MHz, AP 3.3 V |
| Crystal | 40 MHz |
| Security during inspection | Secure Boot disabled; flash encryption disabled |
| Probe app binary | 515,824 B |

The probe app SHA-256 is
`6e600f21464c9f7fa7e0f6b241cbcc98853bda98fec86e98ddc5df22053b8dcc`.
The 4 MiB application partition remains 88% free after linking the embedded model.

## Runtime memory capacity

These values describe memory available to the ESP-IDF capability allocator after the
runtime and probe firmware have initialized. They are more useful for deployment
planning than the nominal on-chip SRAM number alone.

| Heap region at probe start | Total allocatable | Free | Largest block |
|---|---:|---:|---:|
| Internal 8-bit SRAM | 466,651 B | 383,063 B | 253,952 B |
| DMA-capable internal SRAM | 458,739 B | 375,531 B | 253,952 B |
| External PSRAM | 8,388,608 B | 8,385,720 B | 8,257,536 B |

During the internal-arena inference run, minimum free internal heap reached 219,003 B.
The lifetime decrease from the boot snapshot was 164,060 B, consistent with the
163,840-byte arena reservation plus allocator overhead. The arena was released after
the experiment and free heap returned to 382,847 B, so the probe found no persistent
model-allocation leak.

The PSRAM lifetime minimum was 7,861,424 B. That high-water mark comes from the
256 KiB source and destination buffers in the sustained copy test, not from the model
alone. The model's PSRAM arena reservation was 163,840 B and was also fully released.

## Memory bandwidth

| Copy path | Working set | Measured throughput |
|---|---:|---:|
| Internal -> internal | 32 KiB x 256 | 363.372 MiB/s |
| PSRAM -> PSRAM | 256 KiB x 128 | 21.300 MiB/s |
| Internal -> PSRAM | 32 KiB x 256 | 358.938 MiB/s |
| PSRAM -> internal | 32 KiB x 256 | 358.857 MiB/s |

The 256 KiB PSRAM-to-PSRAM result is the useful sustained external-memory number. The
32 KiB cross-region tests fit in the S3 cache and therefore describe cached transfer
behavior; they must not be interpreted as sustained PSRAM bus bandwidth. Against the
older board's saved measurements, internal copy throughput is about 2.0x higher and
sustained PSRAM copy throughput is about 4.5x higher.

## Fast80 model contract and memory

| Item | Value |
|---|---:|
| Model SHA-256 | `70eb89724ae7f5f84097922e481ba8c61946ef7219fdbeddabc81a612ae7691d` |
| TFLite size | 167,976 B |
| Input | `[1, 80, 80, 3]`, INT8, batch 1 |
| Input bytes | 19,200 B |
| Input quantization | scale 1, zero point -128 |
| Output | `[1, 1]`, INT8 |
| Output quantization | scale 0.00390625, zero point -128 |
| Dense MACs | 3,993,536 |
| Parameters | 111,793 |
| Arena reservation | 163,840 B |
| Arena actually used | 83,420 B |
| Reservation margin | 80,420 B |
| Offline fused live-activation peak | 38,400 B |

The S3 runtime uses 14,352 B more arena than the earlier ESP32 profile (83,420 B
versus 69,068 B). This does not indicate a larger neural network: the FlatBuffer hash
is identical. Target-specific ESP-NN scratch buffers and the newer TFLM component
account for runtime-dependent arena usage. Arena sizing must therefore be validated on
each target rather than copied from the ESP32 build.

## Batch-one inference latency

The robust figures below use five warm-ups and 100 measured invocations. A one-tick
task yield occurs only after each timed invocation to keep the RTOS watchdog healthy.

| Arena placement | Mean | Min | Median | P95 | Max | Model-only rate | Effective compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal SRAM | 67.105 ms | 67.087 | 67.107 | 67.119 | 67.133 | 14.90 infer/s | 59.51 MMAC/s |
| Octal PSRAM | 73.478 ms | 73.453 | 73.477 | 73.494 | 73.510 | 13.61 infer/s | 54.35 MMAC/s |

The very narrow min-to-max intervals show stable model execution when camera and Wi-Fi
work are absent. Internal placement saves 6.373 ms per inference. It is feasible on
this probe because the 160 KiB reservation is below the 253,952-byte boot-time largest
internal block. A complete camera application may reduce that largest block, so the
allocation must be rechecked after camera, networking, and frame buffers initialize.

## Per-operator findings

The model has 26 fused operators. The largest individual costs are:

| Rank | Operator index/type | Internal arena | PSRAM arena |
|---:|---|---:|---:|
| 1 | 1 `ADD` | 17.914 ms | 18.086 ms |
| 2 | 2 `CONV_2D` | 8.259 ms | 8.351 ms |
| 3 | 0 `MUL` | 7.758 ms | 7.941 ms |
| 4 | 4 `CONV_2D` | 3.676 ms | 4.270 ms |
| 5 | 3 `DEPTHWISE_CONV_2D` | 3.080 ms | 3.839 ms |

The leading `MUL` and `ADD` together consume 25.672 ms internally, or about 38.2% of
total Invoke time. Before pruning more convolution channels, the export/preprocessing
path should be inspected to determine whether these full-input elementwise operations
can be algebraically folded into input quantization without changing predictions.
This is a measurement-led optimization candidate, not yet a claim that the operators
are removable.

The placement gain is concentrated in memory-sensitive depthwise and early
convolution layers. `MEAN`, `FULLY_CONNECTED`, and `LOGISTIC` show almost no benefit
from moving the arena internally. The complete 52-row two-placement profile is in
`operator_profile.csv`.

## What remains unmeasured

The board's exact product name, camera sensor, camera GPIO map, and LED GPIOs are not
known. Therefore this run deliberately did not initialize a camera or flash LED.
Camera resolution, capture FPS, JPEG/RGB conversion time, camera DMA use, streaming
bandwidth, Wi-Fi throughput, full-pipeline FPS, power, and temperature remain open.
Those are board-level measurements and cannot be inferred safely from the S3 SoC or
from the ESP32-CAM pinout.

Once the S3 board/camera schematic is identified, the next physical experiment should
initialize camera and Wi-Fi first, log the reduced largest internal block, attempt the
83,420-byte used arena with a conservative margin, and then measure capture,
preprocessing, Invoke, response logic, and stream delivery separately.

## Reproducibility and recovery

- Probe source: `firmware/esp32_s3_capacity_probe/`
- Capture helper: `scripts/capture_s3_capacity.py`
- Parser: `scripts/parse_s3_capacity_profile.py`
- Structured summary: `device_profile_summary.json`
- Memory snapshots: `memory_profile.csv`
- Bandwidth samples: `bandwidth_profile.csv`
- Per-operator measurements: `operator_profile.csv`
- Raw serial capture: `raw_serial.log` (local/ignored)
- Original populated-flash backup: `firmware/original_populated_flash_backup.bin`
  (local/ignored), SHA-256
  `6a2127b8926ac7d6bd9de6d01efaa259e6735d10b87b95f1698e65744b1c6628`

The backup covers `0x00000000..0x00437fff`, which contains every populated partition
reported by the original table: boot/config data, the 4,000 KiB factory app, and the
256 KiB model partition. The full 16 MiB read was intentionally avoided after the UART
bridge proved unstable at 460800 baud; copying the populated range at 115200 completed
and was hashed successfully.

To restore it, substitute the current UART device for `<PORT>`:

```bash
python -m esptool --chip esp32s3 --port <PORT> --baud 115200 \
  write_flash 0x0 \
  artifacts/device_profiles/esp32_s3_capacity/firmware/original_populated_flash_backup.bin
```
