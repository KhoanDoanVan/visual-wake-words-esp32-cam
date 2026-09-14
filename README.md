# Visual Wake Words for ESP32-CAM

Notebook-first pipeline for training a **person / no-person** Visual Wake Word model on real MS COCO 2017 images and exporting it as a full-INT8 TensorFlow Lite Micro model for an AI-Thinker ESP32-CAM.

> Version 1 baseline: training, evaluation, INT8 export, and ESP-IDF compilation are complete. Physical-board validation is the next step.

## Version 1 result

| Item | Result |
|---|---:|
| Dataset | 12,000 train / 2,000 validation / 2,000 held-out test images |
| Model | Tiny MobileNetV1, width multiplier 0.25 |
| Input | 96 × 96 × 3 RGB, INT8 |
| Parameters | 111,793 |
| Pre-optimization compute / sparsity | 5.616M MACs/image; 0% exact-zero kernels |
| Float32 structural memory | 436.69 KiB weights; 288 KiB peak live activations |
| Training | 35 epochs; best validation PR-AUC 0.7843 |
| Test threshold | 0.34, selected on validation data |
| Test accuracy | 69.10% |
| Test precision / recall | 64.49% / 82.36% |
| Test F1 / specificity | 72.34% / 56.33% |
| Test ROC-AUC / PR-AUC | 0.8009 / 0.8034 |
| INT8 model size | 167,976 bytes (164.0 KiB) |
| Float–INT8 parity | MAE 0.0090; 98% decision agreement on 100 images |

The float32 memory row is a graph-level baseline, not a measured TFLite Micro tensor-arena requirement.

The ESP-IDF firmware also compiles successfully: 588,032-byte application binary, 24,900 bytes static DRAM, and 67,386 bytes IRAM. The tensor arena and camera buffers use approximately 454 KiB of external PSRAM. See [the firmware build report](firmware/esp32_cam_vww/BUILD_REPORT.md) for details.

The test metrics above use threshold `0.34`. Firmware starts at the more conservative `0.50`; it must be recalibrated using images captured by the physical camera.

## Notebook pipeline

| Notebook | Task |
|---|---|
| `00_project_setup.ipynb` | Environment and experiment contract |
| `01_data_raw.ipynb` | Download raw COCO annotations |
| `02_eda.ipynb` | Exploratory data analysis |
| `03_data_extraction.ipynb` | Create VWW labels and download selected images |
| `04_data_preprocessing.ipynb` | Integrity checks and input pipeline |
| `05_training.ipynb` | Train the compact model |
| `06_evaluation.ipynb` | Evaluate threshold, calibration, and error slices |
| `07_model_profiling.ipynb` | Profile layers, sparsity, MACs, and peak memory before optimization |
| `08_model_export.ipynb` | Export and validate full-INT8 TFLite |
| `09_report_and_deployment.ipynb` | Model card and deployment gates |

## Run the pipeline

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
jupyter lab
```

Run the notebooks in numeric order. Experiment settings are in `configs/base.yaml`.

## Build and flash

The current firmware assumes an AI-Thinker ESP32-CAM with OV2640 and 4 MB PSRAM.

```bash
source scripts/activate_esp_idf.sh
python scripts/verify_firmware_assets.py
./scripts/build_esp32_firmware.sh
./scripts/flash_esp32_firmware.sh /dev/cu.YOUR_SERIAL_PORT --monitor
```

See [the firmware guide](firmware/esp32_cam_vww/README.md) for wiring and boot-mode instructions.

## Important limitation

This is a COCO-domain baseline, not a finished product. Device latency, peak memory, camera preprocessing parity, and accuracy on real ESP32-CAM frames remain to be measured.
