#!/usr/bin/env python3
"""Convert a TFLite FlatBuffer into the firmware's model_data.h contract."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("header", type=Path)
    args = parser.parse_args()
    payload = args.model.read_bytes()
    rows = [
        "  " + ", ".join(f"0x{value:02x}" for value in payload[offset : offset + 12]) + ","
        for offset in range(0, len(payload), 12)
    ]
    text = (
        "#pragma once\n\n#include <cstdint>\n\n"
        "alignas(16) const unsigned char g_vww_model_data[] = {\n"
        + "\n".join(rows)
        + "\n};\n"
        + f"const unsigned int g_vww_model_data_len = {len(payload)};\n"
    )
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
