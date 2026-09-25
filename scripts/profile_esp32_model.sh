#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 MODEL.tflite INPUT_SIZE VARIANT /dev/cu.SERIAL" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/firmware/esp32_cam_vww"
MODEL_PATH="$1"
INPUT_SIZE="$2"
VARIANT="$3"
PORT="$4"
OUTPUT_DIR="$PROJECT_ROOT/artifacts/device_profiles/$VARIANT"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model: $MODEL_PATH" >&2
  exit 2
fi
MODEL_DIR="$(cd "$(dirname "$MODEL_PATH")" && pwd)"
MODEL_PATH="$MODEL_DIR/$(basename "$MODEL_PATH")"
if [[ -z "${IDF_PATH:-}" ]] || ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF is not active. Run: source scripts/activate_esp_idf.sh" >&2
  exit 2
fi

if [[ -n "${VWW_PROFILE_STAGE_DIR:-}" ]]; then
  STAGE_DIR="$VWW_PROFILE_STAGE_DIR"
  mkdir -p "$STAGE_DIR"
  REMOVE_STAGE_DIR=0
else
  STAGE_DIR="$(mktemp -d /private/tmp/vww-device-profile.XXXXXX)"
  REMOVE_STAGE_DIR=1
fi
cleanup() {
  if [[ "$REMOVE_STAGE_DIR" -eq 1 ]]; then
    rm -rf -- "$STAGE_DIR"
  fi
}
trap cleanup EXIT

rsync -a \
  --exclude build \
  --exclude '/dist/' \
  --exclude managed_components \
  --exclude sdkconfig \
  "$SOURCE_DIR/" "$STAGE_DIR/"

python "$SCRIPT_DIR/embed_tflite.py" "$MODEL_PATH" "$STAGE_DIR/include/model_data.h"
pushd "$STAGE_DIR" >/dev/null
if [[ ! -f sdkconfig ]]; then
  idf.py set-target esp32
fi
idf.py -D VWW_INPUT_SIZE="$INPUT_SIZE" -D VWW_MODEL_VARIANT="$VARIANT" \
  -D VWW_PERSON_THRESHOLD="${VWW_PERSON_THRESHOLD_OVERRIDE:-0.275f}" \
  -D VWW_FLASH_LED_ENABLED=0 build
python -m esptool --chip esp32 merge_bin \
  --flash_mode dio --flash_freq 40m --flash_size 4MB \
  -o merged.bin \
  0x1000 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x10000 build/esp32_cam_vww.bin
python -m esptool --chip esp32 --port "$PORT" --baud 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_freq 40m --flash_size 4MB --verify 0x0 merged.bin
popd >/dev/null

mkdir -p "$OUTPUT_DIR/firmware"
cp "$STAGE_DIR/build/esp32_cam_vww.bin" "$OUTPUT_DIR/firmware/esp32_cam_vww.bin"
cp "$STAGE_DIR/build/esp32_cam_vww.elf" "$OUTPUT_DIR/firmware/esp32_cam_vww.elf"
cp "$STAGE_DIR/merged.bin" "$OUTPUT_DIR/firmware/esp32_cam_vww_merged.bin"
python "$SCRIPT_DIR/capture_serial_profile.py" \
  --port "$PORT" --output "$OUTPUT_DIR/device_serial.log"
"$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate_device_profile_reports.py" \
  single --serial-log "$OUTPUT_DIR/device_serial.log" --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR"

echo "Device profile complete: $OUTPUT_DIR"
