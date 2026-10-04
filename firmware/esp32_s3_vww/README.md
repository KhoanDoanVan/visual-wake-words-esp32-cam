# ESP32-S3 120x120 VWW deployment

This firmware runs the frozen full-INT8 Student-120 hard-control model on the
physically verified ESP32-S3-EYE-style OV3660 board.

## Deployed contract

| Item | Value |
|---|---:|
| Model | `student_120_hard_control_int8.tflite` |
| Model SHA-256 | `99b30524db595e90a939274bfac66977694375657bbaeb84217fcdf2e9b13dc0` |
| Model input | 1x120x120x3 INT8 RGB |
| Camera capture | 320x240 QVGA JPEG |
| Output | INT8 person probability |
| Threshold | 0.275, selected on validation data |
| Temporal rule | 2 positive votes in the latest 3 frames |
| Activation rule | motion gates only the dormant-to-person transition |
| Hardware indicator | GPIO3 heartbeat for no-person; steady ON for person |

The 320x240 source is decoded to RGB and bilinearly resized to 120x120. It does
not reuse the older ESP32-CAM gamma/chroma transform; the input follows the
training pipeline's RGB 0..255 contract.

GPIO3 is the status LED defined by Espressif's official ESP32-S3-EYE BSP. A
short pulse approximately every two seconds means the camera/inference loop is
running with no stable person. Steady ON means a person passed the 2-of-3
temporal decision. This board has no verified high-power white illumination
flash. GPIO4 and GPIO5 are camera SCCB pins and must not be driven as LEDs.

## Dashboard

Join Wi-Fi `VWW-Camera` with password `visualwake`, then open
`http://192.168.4.1`. The page shows the QVGA stream, raw person probability,
stabilized state, threshold, motion gate, brightness, and inference latency.

## Build and flash

```bash
source scripts/activate_esp_idf.sh
idf.py -C firmware/esp32_s3_vww set-target esp32s3
idf.py -C firmware/esp32_s3_vww build
idf.py -C firmware/esp32_s3_vww -p /dev/cu.YOUR_PORT -b 460800 flash monitor
```

If the board uses native USB, hold BOOT and tap RESET when `Connecting...`
appears, then release BOOT. Exit the monitor with `Ctrl+]`.

At boot, verify these lines before interpreting predictions:

- OV3660 initialization and QVGA/120x120 contract;
- `AllocateTensors` success and reported arena use;
- input scale 1.0 and zero point -128;
- camera color-bar self-test pass;
- live `p=...` records and GPIO3 wake transitions.
