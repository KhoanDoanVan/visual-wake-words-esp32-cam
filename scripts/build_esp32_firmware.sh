#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/firmware/esp32_cam_vww"
DIST_DIR="$SOURCE_DIR/dist"

if [[ -z "${IDF_PATH:-}" ]] || ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF is not active. First run: source /path/to/esp-idf/export.sh" >&2
  exit 2
fi

python "$SCRIPT_DIR/verify_firmware_assets.py"

# ESP-IDF rejects spaces in project paths. Build from a disposable no-space
# staging directory, then copy only flashable/reviewable outputs back.
STAGE_DIR="$(mktemp -d /private/tmp/vww-esp32-build.XXXXXX)"
BUILD_SUCCEEDED=0
cleanup() {
  if [[ "$BUILD_SUCCEEDED" -eq 1 ]]; then
    rm -rf -- "$STAGE_DIR"
  else
    echo "Build failed; retained incremental build at $STAGE_DIR" >&2
  fi
}
trap cleanup EXIT

rsync -a \
  --exclude build \
  --exclude dist \
  --exclude managed_components \
  --exclude sdkconfig \
  "$SOURCE_DIR/" "$STAGE_DIR/"

pushd "$STAGE_DIR" >/dev/null
idf.py set-target esp32
idf.py build
idf.py size

mkdir -p "$DIST_DIR"
cp build/bootloader/bootloader.bin "$DIST_DIR/bootloader.bin"
cp build/partition_table/partition-table.bin "$DIST_DIR/partition-table.bin"
cp build/esp32_cam_vww.bin "$DIST_DIR/esp32_cam_vww.bin"
cp build/esp32_cam_vww.elf "$DIST_DIR/esp32_cam_vww.elf"
cp build/flasher_args.json "$DIST_DIR/flasher_args.json"
cp build/flash_args "$DIST_DIR/flash_args"
cp build/project_description.json "$DIST_DIR/project_description.json"
cp dependencies.lock "$SOURCE_DIR/dependencies.lock"

python -m esptool --chip esp32 merge_bin \
  --flash_mode dio \
  --flash_freq 40m \
  --flash_size 4MB \
  -o "$DIST_DIR/esp32_cam_vww_merged.bin" \
  0x1000 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x10000 build/esp32_cam_vww.bin
popd >/dev/null
BUILD_SUCCEEDED=1

echo
echo "Build complete: $DIST_DIR/esp32_cam_vww_merged.bin"
echo "Next: $SCRIPT_DIR/flash_esp32_firmware.sh /dev/cu.YOUR_SERIAL_PORT"
