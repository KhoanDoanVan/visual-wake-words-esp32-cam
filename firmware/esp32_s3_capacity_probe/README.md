# ESP32-S3 capacity probe

This isolated firmware measures the connected ESP32-S3 without assuming a camera pin
map. It does not initialize a camera, Wi-Fi, or any LED/flash GPIO.

It records:

- CPU, flash, PSRAM, and allocatable heap capacity;
- internal and external memory copy throughput;
- Fast80 tensor contract and actual arena use;
- batch-one per-operator latency;
- 100-sample mean/min/median/p95/max Invoke latency; and
- output parity between PSRAM and internal-SRAM arenas.

The dependency lock fixes ESP-IDF 5.3, esp-tflite-micro 1.4.1, and ESP-NN 1.4.0.
Before a clean build, generate the ignored model header from the frozen artifact:

```bash
python3 scripts/embed_tflite.py \
  artifacts/fast_80/models/vww_mobilenetv1_80_int8.tflite \
  firmware/esp32_s3_capacity_probe/include/model_data.h
```

Then build for `esp32s3`, flash the UART port, and capture one complete run. If the
workspace path contains spaces and the local ESP-IDF/CMake combination rejects it,
stage this firmware directory under `/private/tmp` before building.

```bash
source /path/to/esp-idf/export.sh
idf.py -C firmware/esp32_s3_capacity_probe set-target esp32s3 build
idf.py -C firmware/esp32_s3_capacity_probe -p <PORT> -b 115200 flash

python3 scripts/capture_s3_capacity.py \
  --port <PORT> \
  --output artifacts/device_profiles/esp32_s3_capacity/raw_serial.log

python3 scripts/parse_s3_capacity_profile.py \
  --log artifacts/device_profiles/esp32_s3_capacity/raw_serial.log \
  --output-dir artifacts/device_profiles/esp32_s3_capacity \
  --model artifacts/fast_80/models/vww_mobilenetv1_80_int8.tflite \
  --firmware artifacts/device_profiles/esp32_s3_capacity/firmware/esp32_s3_capacity_probe.bin
```

The measured results and recovery details are in
`artifacts/device_profiles/esp32_s3_capacity/DEVICE_CAPACITY_REPORT.md`.
