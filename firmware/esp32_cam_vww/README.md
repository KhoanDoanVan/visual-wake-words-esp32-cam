# ESP32-CAM Visual Wake Word firmware

This project runs the exported 96×96 full-INT8 person detector on an **AI-Thinker ESP32-CAM with OV2640 and 4 MB PSRAM**. It captures 160×120 RGB565 frames, reproduces TensorFlow's half-pixel bilinear resize, invokes TFLite Micro, debounces the score, and drives the board's active-low red LED on GPIO33.

## Hardware assumption

Check the marking and pinout before applying power. The committed map is for the common AI-Thinker module, not every board sold as “ESP32-CAM.” This firmware requires working PSRAM; it deliberately stops at boot if PSRAM initialization fails.

For flashing with a 3.3 V USB-to-UART adapter:

| USB-to-UART | ESP32-CAM |
|---|---|
| TX (3.3 V logic) | U0R / GPIO3 |
| RX (3.3 V logic) | U0T / GPIO1 |
| GND | GND |
| stable 5 V supply | 5V |

Do not feed 5 V logic into the serial pins. Many USB-to-UART adapters cannot supply the camera's peak current; use a stable 5 V supply and a shared ground. Connect **GPIO0 to GND only while flashing**, reset/power-cycle the board, flash, disconnect GPIO0 from GND, and reset again to boot normally.

## Toolchain

Use ESP-IDF **v5.3 through v5.5.x** (the project is compiled in CI/preflight with v5.3). Install it outside any path containing spaces, then activate its shell environment:

```bash
source /path/without/spaces/esp-idf/export.sh
idf.py --version
```

On the machine used for this build, the existing IDF checkout needed its already-installed Python 3.10 environment selected explicitly. The repository includes a convenience activator for that setup:

```bash
source scripts/activate_esp_idf.sh
idf.py --version
```

Override `ESP_IDF_DIR` or `ESP_IDF_PYTHON_ENV` before sourcing it if your installation lives elsewhere.

The component manager resolves the pinned compatible ranges in `main/idf_component.yml`:

- `espressif/esp32-camera` 2.1.x
- `espressif/esp-tflite-micro` 1.4.x

## Preflight and build

Run from the repository root:

```bash
python scripts/verify_firmware_assets.py
./scripts/build_esp32_firmware.sh
```

The repository itself contains spaces, so the build helper copies the project to a temporary no-space path. It returns a merged 4 MB flash image and ELF file in `firmware/esp32_cam_vww/dist/`.

## Flash and monitor

Find your port on macOS, put the board into download mode, and flash:

```bash
ls /dev/cu.*
./scripts/flash_esp32_firmware.sh /dev/cu.usbserial-XXXX --monitor
```

If automatic reset is unavailable, press RESET just after the flashing command starts. Exit the serial monitor with `Ctrl+]`.

Healthy startup logs include:

```text
VWW startup: model=167976 bytes threshold=0.50 debounce=4/5
Input INT8 scale=1.000000 zero=-128 bytes=27648
Output INT8 scale=0.003906 zero=-128 bytes=1
ready: internal_free=... psram_free=...
p=... wake=... inference=...ms
```

The first device test should verify camera initialization, `AllocateTensors`, free memory, inference latency, and LED behavior. The 350 KiB tensor arena is a conservative PSRAM allocation, not a claim that the network fits in the ESP32's 256 KiB internal SRAM.

See `BUILD_REPORT.md` for the successful compile's exact flash/DRAM/IRAM usage, dependency versions, artifact hashes, and remaining physical-device gates.

## Runtime tuning

Edit `main/app_config.h` to tune the score threshold, temporal debounce, frame interval, or output GPIO. The current threshold is 0.50: on the held-out host test set it traded some recall for materially fewer false positives than the validation-selected 0.34 threshold. Recalibrate on frames captured by the actual ESP32-CAM before treating the result as deployment-ready.

If the red LED never changes, remember it is active-low. If you need a signal on an exposed pin, select a pin that is genuinely free on your board and is not used by the camera, serial console, flash boot strap, microSD, or flash LED.
