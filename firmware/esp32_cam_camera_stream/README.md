# ESP32-CAM OV3660 maximum-resolution stream

Diagnostic firmware for the classic AI-Thinker ESP32-CAM. It captures the
installed OV3660 at QXGA (2048 x 1536) in JPEG mode and exposes a local web
dashboard with live stream and memory/capture telemetry.

## Connect

1. Join Wi-Fi `ESP32-CAM-Max` with password `visualwake`.
2. Open `http://192.168.4.1/`.

The dashboard is served on port 80. MJPEG is served on port 81 at `/stream`;
`/capture` downloads a full-resolution JPEG and `/status` returns JSON.

GPIO 4 is forced low and held low, so the white flash LED remains disabled.
QXGA is intended for camera-capacity inspection and dataset capture, not VWW
inference. Use a smaller capture frame and resize/crop to the model input for
real-time inference.

## Measured on the connected board

- MCU: ESP32-D0WD-V3 revision 3.1, dual core, 240 MHz
- Flash: 4 MB
- PSRAM device: 8 MB detected; 4 MB is addressable/mapped by the classic ESP32
- Sensor: OV3660, PID `0x3660`
- Stable maximum stream frame: QXGA, 2048 x 1536 JPEG
- XCLK / pixel clock: 10 MHz / 5 MHz
- Steady capture: about 341.6 ms per frame, 2.77-2.78 fps
- Typical JPEG at quality 12 in the measured scene: about 86 KB
- Free after camera + latest-frame allocation: about 2.65 MB PSRAM and 71 KB internal RAM
- Camera framebuffer allocation: 629,145 bytes in PSRAM
