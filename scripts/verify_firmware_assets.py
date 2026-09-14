#!/usr/bin/env python3
"""Offline preflight checks for the firmware/model handoff."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/models/vww_mobilenetv1_96_int8.tflite"
HEADER = ROOT / "firmware/esp32_cam_vww/include/model_data.h"
EXPORT_INFO = ROOT / "artifacts/reports/export_info.json"
APP_CONFIG = ROOT / "firmware/esp32_cam_vww/main/app_config.h"


def fail(message: str) -> None:
    raise SystemExit(f"PRE-FLIGHT FAILED: {message}")


for required in (MODEL, HEADER, EXPORT_INFO, APP_CONFIG):
    if not required.is_file():
        fail(f"missing {required.relative_to(ROOT)}")

model_bytes = MODEL.read_bytes()
header_text = HEADER.read_text(encoding="utf-8")
header_bytes = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", header_text))
length_match = re.search(r"g_vww_model_data_len\s*=\s*(\d+)", header_text)
if length_match is None:
    fail("model_data.h has no g_vww_model_data_len declaration")

declared_length = int(length_match.group(1))
if model_bytes != header_bytes:
    fail("model_data.h payload differs from the exported .tflite file")
if declared_length != len(model_bytes):
    fail(f"header declares {declared_length} bytes, actual model is {len(model_bytes)} bytes")

contract = json.loads(EXPORT_INFO.read_text(encoding="utf-8"))
expected_ops = {
    "ADD",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
    "LOGISTIC",
    "MEAN",
    "MUL",
}
if contract["input"] != {
    "name": "serving_default_image:0",
    "shape": [1, 96, 96, 3],
    "dtype": "int8",
    "scale": 1.0,
    "zero_point": -128,
}:
    fail(f"unexpected input contract: {contract['input']}")
if contract["output"]["shape"] != [1, 1] or contract["output"]["dtype"] != "int8":
    fail(f"unexpected output contract: {contract['output']}")
if set(contract["operators"]) != expected_ops:
    fail(f"unexpected operator set: {contract['operators']}")

config_text = APP_CONFIG.read_text(encoding="utf-8")
threshold_match = re.search(r"kPersonThreshold\s*=\s*([0-9.]+)f", config_text)
threshold = float(threshold_match.group(1)) if threshold_match else None
if threshold is None or not 0.0 < threshold < 1.0:
    fail("invalid kPersonThreshold in app_config.h")

print("Firmware preflight: PASS")
print(f"  model: {len(model_bytes):,} bytes ({hashlib.sha256(model_bytes).hexdigest()})")
print(f"  input: {contract['input']['shape']} INT8, scale=1.0, zero_point=-128")
print(
    "  output: "
    f"{contract['output']['shape']} INT8, scale={contract['output']['scale']}, "
    f"zero_point={contract['output']['zero_point']}"
)
print(f"  operators: {', '.join(contract['operators'])}")
print(f"  firmware threshold: {threshold:.2f}")

