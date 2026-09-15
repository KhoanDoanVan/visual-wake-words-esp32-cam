# ESP32-CAM Visual Wake Word firmware

This project runs the exported 80×80 full-INT8 person detector on an **AI-Thinker-pinout ESP32-CAM with PSRAM**. The photographed board has an OV3660 sensor (also confirmed by its runtime probe). It captures 160×120 JPEG frames, decodes Espressif's BGR-ordered conversion output back to model-native RGB, reproduces TensorFlow's half-pixel bilinear resize, applies a fixed-point real-camera color transform, invokes TFLite Micro, gates activation with frame motion, debounces the score, and publishes the camera plus inference telemetry on a device-local dashboard.

| Inference state | Dashboard | Onboard light |
|---|---|---|
| Non-person | Red | Small red LED on (GPIO33, active-low) |
| Person detected | Blue | Small red LED off |

GPIO4 controls the board's bright white camera flash LED. The firmware configures GPIO4 as an output and holds it low at startup and after every inference decision, so it is never used as a status light. The photographed ESP32-CAM and ESP32-CAM-MB do not contain a controllable blue LED; blue person status is therefore shown on the webpage.

## Live camera dashboard

After the firmware boots:

1. Connect a phone or computer to Wi-Fi **`VWW-Camera`** using password **`visualwake`**.
2. Ignore the “no internet” warning and remain connected to that network.
3. Open **`http://192.168.4.1`** in a browser (use `http`, not `https`).

The page shows the live 160×120 camera stream, person probability, the 0.44 post-transform threshold, true model-input brightness, inference latency, frame rate, current-frame classification, frame motion, activation gate, and temporal vote. Its states are:

- **Blue / PERSON**: while dormant, at least 2 of the latest 3 frames were at or above 0.44 and contained meaningful scene motion. Once active, the classifier alone maintains or clears the state, so a person may stand still.
- **Red / NO PERSON**: the model has no stable person detection.
- **Amber / TOO DARK**: mean input brightness is below 30/255, so inference is skipped.
- **CAMERA ERROR**: capture, conversion, self-test, or inference failed.

The page, JSON telemetry, and visible camera frames now use port 80. The dashboard requests
`/frame` only when `/status` reports a new frame sequence, avoiding cross-port failures and
stale long-lived MJPEG connections. A legacy MJPEG endpoint remains available at
`http://192.168.4.1:81/stream`, but the webpage does not depend on it. The dashboard is hosted
by the ESP32 itself and does not require internet access.

The model is invoked once for every live camera frame (batch size 1). The JPEG and its inference
telemetry are committed together only after that invocation finishes, so a newly displayed image
cannot be paired with the previous frame's probability. The stable state uses a 2-of-3 sliding
vote. A static model-positive background cannot contribute activation votes; the separate
**Current frame**, **Frame motion**, **Activation gate**, and **Temporal vote** fields explain
confirmation, suppression, and release behavior.

## Real-camera preprocessing

The deployed model and its training dataset are unchanged. During the existing bilinear resize,
firmware reduces chroma by 50% around integer BT.601 luminance and applies a gamma-1.2 lookup
table. This corrects the low-color OV3660 stream using a 256-byte flash LUT, integer arithmetic,
and no additional image buffer. Evaluation on the supplied screen recording showed the labeled,
non-transition INT8 score ranges move from overlapping (person 0.301-0.574; non-person up to
0.688) to separated (person 0.484-0.602; non-person up to 0.434). These figures are calibration
evidence for that recording, not a replacement for testing in additional lighting and viewpoints.

An inactive wake state also requires mean absolute frame change of at least 2.0 input levels.
The motion calculation reuses the existing 19,200-byte previous-input PSRAM buffer. It does not
invoke a second network and does not increase the tensor arena.

If telemetry updates but the image is blank, perform a hard refresh so the browser discards
the earlier page that referenced port 81. The waiting panel now reports frame-request failures
and retries automatically.

## Hardware assumption

Check the marking and pinout before applying power. The committed map is for the common AI-Thinker module, not every board sold as “ESP32-CAM.” The connected module reports 8 MB physical PSRAM with 4 MB mapped by this ESP32 build; the firmware requires working PSRAM and deliberately stops at boot if initialization fails.

For an OV3660, startup also applies Espressif's reference corrections: vertical flip enabled, brightness `+1`, and saturation `-2`. These settings and the BGR-to-RGB correction must remain aligned with the RGB preprocessing used for training.

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
VWW startup: model=167976 bytes threshold=0.27 debounce=2/3
Input INT8 scale=1.000000 zero=-128 bytes=19200
Output INT8 scale=0.003906 zero=-128 bytes=1
Dashboard ready: Wi-Fi 'VWW-Camera' password 'visualwake', open http://192.168.4.1
Camera self-test passed (color-bar mean ...)
ready: internal_free=... psram_free=...
p=... wake=... inference=...ms
```

The first device test should verify camera initialization, `AllocateTensors`, free memory, inference latency, and LED behavior. The 350 KiB tensor arena is a conservative PSRAM allocation, not a claim that the network fits in the ESP32's 256 KiB internal SRAM.

See `BUILD_REPORT.md` for build and deployment verification. See
`HARDWARE_CAPACITY_REPORT.md` for the measured CPU, memory, camera, bandwidth,
batch-1 pipeline, FPS, and model-design limits of the connected board.

## Real-device operator profiling

The firmware passes a lightweight profiler to TFLite Micro and records each
fused FlatBuffer operator by execution index. It skips two warm-up invocations,
aggregates 20 batch-size-1 invocations, emits one machine-readable
`VWW_PROFILE` block over serial, and then becomes a near-no-op. The accumulator
uses static storage so it does not consume the small main-task stack.

Build, flash, capture, and generate a report for any compatible model with:

```bash
source scripts/activate_esp_idf.sh
scripts/profile_esp32_model.sh MODEL.tflite INPUT_SIZE VARIANT /dev/cu.YOUR_SERIAL_PORT
```

The standard runs use variants `baseline_96` and `fast_80`. The generated CSV
maps the device timing to tensor shapes, constant bytes, activation bytes, and
estimated MACs from that exact TFLite FlatBuffer. It also reconstructs tensor
lifetimes for every fused operator, reports live activation RAM before/during/after
execution, and combines that estimate with measured TFLM arena usage and board heap
snapshots. Constants are reported as flash storage and are not counted as SRAM/PSRAM.

## Runtime tuning

Edit `main/app_config.h` to tune the post-transform score threshold, motion threshold, temporal debounce, frame interval, or output GPIO. The current 0.44 score threshold and 2.0 motion threshold were selected from the supplied real-device recording; keep the score threshold aligned with the preprocessing transform. The older 0.27 value was selected before camera-domain correction and must not be reused with the transformed input.

If the red LED never changes, remember it is active-low: red off means the debounced person state is active. GPIO4 is reserved only to keep the white flash off. If you add an external indicator, select a pin that is genuinely free on your board and is not used by the camera, serial console, flash boot strap, microSD, or flash LED, and use a series resistor.
