#!/usr/bin/env python3
"""Capture a complete VWW_PROFILE block from an ESP32 serial port."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.timeout
    lines: list[str] = []
    with serial.Serial(args.port, 115200, timeout=1) as connection:
        while time.monotonic() < deadline:
            raw = connection.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()
            print(line, flush=True)
            lines.append(line)
            if "VWW_PROFILE,END" in line:
                args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raise SystemExit(f"timed out before VWW_PROFILE,END; partial log saved to {args.output}")


if __name__ == "__main__":
    main()
