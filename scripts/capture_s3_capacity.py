#!/usr/bin/env python3
"""Reset an ESP32-S3 and capture one complete S3_CAPACITY report."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    deadline = time.monotonic() + args.timeout
    with serial.Serial(args.port, 115200, timeout=1) as connection:
        # QinHeng bridge: RTS drives EN active-low and DTR drives GPIO0.
        connection.dtr = False
        connection.rts = True
        time.sleep(0.1)
        connection.rts = False
        time.sleep(0.2)
        connection.reset_input_buffer()
        while time.monotonic() < deadline:
            raw = connection.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()
            print(line, flush=True)
            lines.append(line)
            if "S3_CAPACITY,END" in line:
                args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raise SystemExit(f"timed out before S3_CAPACITY,END; partial log saved to {args.output}")


if __name__ == "__main__":
    main()
