# ESP32-CAM firmware build report

Build, flash, and connected-device smoke-test status: **PASS** on 2026-09-15 using ESP-IDF v5.3, Xtensa GCC 13.2.0, and esptool 4.11.0 for target `esp32`.

## Resolved dependencies

| Component | Version |
|---|---:|
| `espressif/esp-tflite-micro` | 1.4.0 |
| `espressif/esp-nn` | 1.3.2 |
| `espressif/esp32-camera` | 2.1.7 |
| `espressif/esp_jpeg` | 1.3.1 |

Exact hashes are recorded in `dependencies.lock`.

## Static memory and flash

| Region | Used | Available to linked image | Usage |
|---|---:|---:|---:|
| DRAM | 49,160 B | 180,736 B | 27.20% |
| IRAM | 111,238 B | 131,072 B | 84.87% |
| Flash code | 751,039 B | — | — |
| Flash data | 360,028 B | — | — |
| Total image content | 1,250,849 B | — | — |
| Application binary | 1,250,960 B | 3,145,728 B partition | 39.77% |

The 167,976-byte TFLite file is compiled into flash data. It is not a 436 KB float32 parameter allocation in internal SRAM.

## Runtime external-RAM budget

The following buffers are deliberately allocated from the module's 4 MB PSRAM:

| Allocation | Budget |
|---|---:|
| TFLite Micro tensor arena | 358,400 B |
| RGB888 conversion scratch | 57,600 B |
| 160×120 JPEG camera framebuffer capacity | 184,320 B plus driver metadata |
| Latest JPEG dashboard frame | 184,320 B |
| Pending JPEG awaiting its matching inference result | 184,320 B |
| Per-connected MJPEG viewer buffer | 184,320 B |
| Same-origin `/frame` response | Exact current JPEG length, transient |
| Previous model input (motion gate and diagnostics) | 19,200 B |

The steady allocations total about 992 KB plus allocator/driver overhead, with another 180 KB
only if the optional legacy MJPEG endpoint is used. The main webpage uses short same-origin
frame responses and allocates only the current JPEG length temporarily. TFLite Micro reports
69,068 B actually used inside the 358,400-byte tensor-arena reservation.

## Produced artifacts

| File | Size | SHA-256 |
|---|---:|---|
| `dist/esp32_cam_vww.bin` | 1,250,960 B | `c67dd7de9a6712f39bc1d10a14efb592082199617cf4217ca2c1521397092ae7` |
| `dist/esp32_cam_vww_merged.bin` | 1,316,496 B | `f69649d76a72db584892f1190fc4d390ebc6a3710483e0fd83a7737efb481185` |
| `dist/bootloader.bin` | 26,752 B | `1973fbf219e0a879d59bb8cda171f506f1857bd69b9c618f8da3b6bbe65c02eb` |
| `dist/partition-table.bin` | 3,072 B | `73c0b5c3e5fcba3a151cc70c453c93dd5f4798899e7f2f8cca76da1f32ffc501` |

Espressif image inspection found a valid checksum and valid SHA validation hash in the application binary.

## Connected-device results

- Flashing and post-write hash verification passed over the ESP32-CAM-MB USB serial adapter.
- The ESP32-D0WD-V3 rev 3.1 reported an 8 MB PSRAM device, mapped 4 MB, and passed the boot memory test.
- The sensor probed as an OV3660 at address `0x3c`; JPEG camera initialization and a 184,320-byte PSRAM framebuffer allocation passed.
- OV3660 vertical flip, brightness `+1`, and saturation `-2` corrections were accepted by the sensor.
- OV3660 AEC, AEC2, AGC, and AWB were enabled with AE `+1` and a 64× gain ceiling.
- Firmware corrects the ESP32 camera converter's BGR byte order to the RGB channel order used during training.
- TFLite Micro accepted the expected 19,200-byte INT8 input and one-byte INT8 output tensor contracts.
- The sensor-generated color-bar self-test passed with post-transform mean brightness 124.6/255, proving the camera digital path and RGB preprocessing are operating.
- The `VWW-Camera` WPA2 access point and both HTTP servers started at `192.168.4.1`.
  The webpage now obtains images from the same-origin `/frame` endpoint on port 80.
- Ready-state memory was 66,811 bytes free internal RAM and 3,202,860 bytes free mapped PSRAM.
- The final device smoke test continuously captured 1,830–1,850-byte JPEG frames, completed
  inference, published dashboard frames, and emitted a complete 26-operator profile without
  camera, allocation, or watchdog errors.
- With operator profiling enabled, batch-1 inference was 414.5–422.4 ms (416.8 ms mean) and the observed full frame period was 597–601 ms (1.66–1.68 fps). JPEG decode, RGB correction, resize, fixed-point chroma reduction, gamma lookup, and INT8 conversion took 59.8–61.3 ms. See `HARDWARE_CAPACITY_REPORT.md` for the broader benchmark.
- With inference disabled for the measurement, 20 camera captures completed at 6.94 fps and averaged 1,838.6 bytes per JPEG in the current scene.
- Startup copy tests measured 181.55 MiB/s for internal SRAM and 4.71 MiB/s for mapped 40 MHz PSRAM.
- The model is invoked once per captured live frame. The dashboard atomically commits each JPEG with that frame's inference result and polls telemetry every 400 ms.
- The model binary and training dataset are unchanged. The deployment input now applies 50% chroma reduction around BT.601 luma followed by a gamma-1.2 256-byte LUT, with a matched score threshold of 0.44.
- The temporal rule remains 2-of-3 without an initial full-window wait. While dormant, votes additionally require frame motion of at least 2.0 input levels. Once awake, the model alone maintains or releases the state.
- In the final static-scene smoke test, raw scores remained high at 0.508–0.586, but measured motion was only 0.4–0.7 and `wake=0` throughout. This directly verifies static false-positive suppression on the connected board.
- GPIO33 red was configured for non-person. GPIO4 was configured as an output and repeatedly forced low; the white flash is disabled. Person state is blue on the dashboard.

## Remaining validation gates

- Exercise a person entering, remaining still, and leaving in several intended lighting conditions. Confirm blue/red dashboard transitions; the onboard red LED should turn off for person and the white flash must remain off.
- Confirm that typical person entry produces motion above 2.0 on at least two of three frames. If it does not, lower only `kActivationMotionThreshold`; keep the 0.44 score threshold aligned with the camera transform.
- Repeat the recording-style validation from additional viewpoints before treating the single-recording calibration ranges as production accuracy measurements.

## Optimized model validation

The deployed model retains the VWW MobileNetV1-style depthwise-separable chain and reduces its input from 96x96 to 80x80. It has 111,793 parameters, 3,993,536 estimated MACs, and a 51,200-byte peak live INT8 activation estimate. This cuts estimated MACs by 29% and peak activation by 31%; the model file remains 167,976 bytes because resolution changes activations rather than weight count.

| INT8 pipeline at threshold 0.27 | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Held-out COCO test (2,000 images) | 65.75% | 60.10% | 89.81% | 72.01% | 78.73% |

The previous 96x96 baseline F1 was 72.34%, so the optimized model retained F1 within 0.33 percentage points while increasing recall. Real labeled OV3660 frames are still required to quantify exposure, optics, background, and placement shift.
