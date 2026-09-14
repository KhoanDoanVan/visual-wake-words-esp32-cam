#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/firmware/esp32_cam_vww/dist"
IMAGE="$DIST_DIR/esp32_cam_vww_merged.bin"
PORT="${1:-}"

if [[ -z "$PORT" ]]; then
  echo "Usage: $0 /dev/cu.YOUR_SERIAL_PORT [--monitor]" >&2
  echo "Available serial ports:" >&2
  find /dev -maxdepth 1 -name 'cu.*' -print 2>/dev/null >&2 || true
  exit 2
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Missing $IMAGE; run scripts/build_esp32_firmware.sh first." >&2
  exit 2
fi
if [[ -z "${IDF_PATH:-}" ]] || ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF is not active. First run: source /path/to/esp-idf/export.sh" >&2
  exit 2
fi

python -m esptool \
  --chip esp32 \
  --port "$PORT" \
  --baud 460800 \
  --before default_reset \
  --after hard_reset \
  write_flash \
  --flash_mode dio \
  --flash_freq 40m \
  --flash_size 4MB \
  --verify \
  0x0 "$IMAGE"

echo "Flash complete. Remove GPIO0 from GND, then press RESET."

if [[ "${2:-}" == "--monitor" ]]; then
  python -m esp_idf_monitor \
    --port "$PORT" \
    --baud 115200 \
    "$DIST_DIR/esp32_cam_vww.elf"
fi
