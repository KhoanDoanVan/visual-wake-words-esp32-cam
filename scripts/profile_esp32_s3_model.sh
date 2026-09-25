#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 MODEL.tflite INPUT_SIZE MACS ARENA_BYTES VARIANT /dev/cu.SERIAL" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/firmware/esp32_s3_capacity_probe"
MODEL_PATH="$1"
INPUT_SIZE="$2"
MACS="$3"
ARENA_BYTES="$4"
VARIANT="$5"
PORT="$6"
OUTPUT_DIR="$PROJECT_ROOT/artifacts/device_profiles/$VARIANT/esp32_s3"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model: $MODEL_PATH" >&2
  exit 2
fi
if [[ -z "${IDF_PATH:-}" ]] || ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF is not active. Run: source scripts/activate_esp_idf.sh" >&2
  exit 2
fi

STAGE_DIR="$(mktemp -d /private/tmp/vww-s3-profile.XXXXXX)"
cleanup() { rm -rf -- "$STAGE_DIR"; }
trap cleanup EXIT

rsync -a --exclude build --exclude managed_components --exclude sdkconfig \
  "$SOURCE_DIR/" "$STAGE_DIR/"
python "$SCRIPT_DIR/embed_tflite.py" "$MODEL_PATH" "$STAGE_DIR/include/model_data.h"

pushd "$STAGE_DIR" >/dev/null
idf.py set-target esp32s3
idf.py -D VWW_INPUT_SIZE="$INPUT_SIZE" -D VWW_MODEL_VARIANT="$VARIANT" \
  -D VWW_ARENA_BYTES="$ARENA_BYTES" build
idf.py -p "$PORT" -b 460800 flash
popd >/dev/null

mkdir -p "$OUTPUT_DIR/firmware"
cp "$STAGE_DIR/build/esp32_s3_capacity_probe.bin" "$OUTPUT_DIR/firmware/"
cp "$STAGE_DIR/build/esp32_s3_capacity_probe.elf" "$OUTPUT_DIR/firmware/"
python "$SCRIPT_DIR/capture_s3_capacity.py" --port "$PORT" \
  --output "$OUTPUT_DIR/raw_serial.log"
python "$SCRIPT_DIR/parse_s3_capacity_profile.py" \
  --log "$OUTPUT_DIR/raw_serial.log" --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_PATH" \
  --firmware "$OUTPUT_DIR/firmware/esp32_s3_capacity_probe.bin" --macs "$MACS"

echo "ESP32-S3 device profile complete: $OUTPUT_DIR"
