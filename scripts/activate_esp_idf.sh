#!/usr/bin/env bash
# Source this file; executing it cannot modify the calling shell environment.

VWW_IDF_DIR="${ESP_IDF_DIR:-/Users/${USER}/esp/esp-idf}"
VWW_IDF_PYTHON_ENV="${ESP_IDF_PYTHON_ENV:-/Users/${USER}/.espressif/python_env/idf5.3_py3.10_env}"

if [[ ! -f "$VWW_IDF_DIR/export.sh" ]]; then
  echo "ESP-IDF export script not found: $VWW_IDF_DIR/export.sh" >&2
  echo "Set ESP_IDF_DIR to your ESP-IDF 5.3-5.5.x checkout." >&2
  return 2 2>/dev/null || exit 2
fi
if [[ -x "$VWW_IDF_PYTHON_ENV/bin/python" ]]; then
  export IDF_PYTHON_ENV_PATH="$VWW_IDF_PYTHON_ENV"
fi

source "$VWW_IDF_DIR/export.sh"
