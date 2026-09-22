# ESP32-S3 OV3660 maximum-resolution stream

This firmware streams the physically verified OV3660 at its maximum QXGA resolution
(2048x1536) as MJPEG. It creates a local Wi-Fi access point and does not configure or
touch a flash/illumination GPIO.

## Connect

1. Join Wi-Fi `S3-Camera-Max` with password `visualwake`.
2. Open `http://192.168.4.1/`.

The dashboard is served on port 80. MJPEG is on port 81 at `/stream`; `/capture`
downloads the latest full-resolution JPEG and `/status` returns JSON telemetry.

QXGA is intended for camera inspection, not VWW inference. The Fast80 deployment
should use a smaller capture size and resize to the model's 80x80 input.
