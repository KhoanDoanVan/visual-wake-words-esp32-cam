# ESP32-CAM firmware build report

Build status: **PASS** on 2026-09-14 using ESP-IDF v5.3, Xtensa GCC 13.2.0, and esptool 4.11.0 for target `esp32`.

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
| DRAM | 24,900 B | 180,736 B | 13.78% |
| IRAM | 67,386 B | 131,072 B | 51.41% |
| Flash code | 251,771 B | — | — |
| Flash data | 249,600 B | — | — |
| Total image content | 587,929 B | — | — |
| Application binary | 588,032 B | 3,145,728 B partition | 18.70% |

The 167,976-byte TFLite file is compiled into flash data. It is not a 436 KB float32 parameter allocation in internal SRAM.

## Runtime external-RAM budget

The following buffers are deliberately allocated from the module's 4 MB PSRAM:

| Allocation | Budget |
|---|---:|
| TFLite Micro tensor arena | 358,400 B |
| RGB888 conversion scratch | 57,600 B |
| 160×120 RGB565 camera framebuffer | about 38,400 B plus driver metadata |

These total about 454 KB plus allocator/driver overhead. Actual tensor-arena high-water usage and latency still require the physical board. Startup logs report free/largest internal RAM and PSRAM before allocation and after the interpreter is ready.

## Produced artifacts

| File | Size | SHA-256 |
|---|---:|---|
| `dist/esp32_cam_vww.bin` | 588,032 B | `3bfba706737031a3f1fe739b975f16498f92c904dbb392fea6f57d3c6489a0ad` |
| `dist/esp32_cam_vww_merged.bin` | 653,568 B | `067fc49f3b2e279be3a84aa5b94fc8bb2d79b9888983a763c49b7e6203808dc0` |
| `dist/bootloader.bin` | 26,752 B | `71f85099bb012c01d5447eb0df22d48fcc48107a9beda5f87c206aab280ec9c9` |
| `dist/partition-table.bin` | 3,072 B | `73c0b5c3e5fcba3a151cc70c453c93dd5f4798899e7f2f8cca76da1f32ffc501` |

Espressif image inspection found a valid checksum and valid SHA validation hash in the application binary.

## Remaining physical-board gates

- Confirm the module is the AI-Thinker OV2640 pinout with working 4 MB PSRAM.
- Flash and confirm camera probe and `AllocateTensors` succeed.
- Record measured inference latency and memory logs.
- Compare ESP32-captured-frame probabilities with host preprocessing.
- Recalibrate the 0.50 threshold on device-domain images.

No ESP32 USB serial device was connected during this build, so the firmware has not been flashed or runtime-tested yet.
