# ESP32-S3 camera capability probe

This isolated ESP-IDF diagnostic identifies the connected camera and verifies its
usable JPEG resolutions by capturing three frames at each size. It does not configure
or touch an illumination/flash LED GPIO.

## Physical-device result

Measured on the connected ESP32-S3 on 2026-09-22:

- Sensor: OV3660, PID `0x3660`, SCCB address `0x3c`
- Maximum successful JPEG capture: QXGA, 2048x1536 (3,145,728 pixels)
- QXGA mean compressed size: 92,452 bytes at JPEG quality 12
- QXGA mean blocking capture time: 179,806 us across three samples
- Camera clock: 20 MHz XCLK; the driver reported 10 MHz PCLK at QXGA
- Framebuffer: one 629,145-byte PSRAM allocation
- Camera PSRAM-DMA: disabled for the stable resolution sweep
- Illumination/flash LED: disabled and unassigned

The measured maximum is the still-image capture limit. It is not the recommended VWW
operating resolution. The Fast80 pipeline should capture QQVGA or QVGA JPEG, crop and
resize to 80x80, and keep the model tensor arena in internal SRAM when the complete
camera/Wi-Fi allocation permits it.

## Camera pin map

The successful probe confirms this ESP32-S3-EYE-style mapping:

| Signal | GPIO |
|---|---:|
| SIOD / SIOC | 4 / 5 |
| VSYNC / HREF / PCLK | 6 / 7 / 13 |
| XCLK | 15 |
| D0..D7 | 11, 9, 8, 10, 12, 18, 17, 16 |
| PWDN / RESET | not connected |

## Build and run

```bash
source /Users/doanvankhoan/esp/esp-idf/export.sh
idf.py -C firmware/esp32_s3_camera_probe set-target esp32s3 build
idf.py -C firmware/esp32_s3_camera_probe \
  -p /dev/cu.usbmodem5B8E0492771 -b 460800 flash
idf.py -C firmware/esp32_s3_camera_probe \
  -p /dev/cu.usbmodem5B8E0492771 -b 115200 monitor
```

The serial path is machine-specific and may change after reconnecting the board.
