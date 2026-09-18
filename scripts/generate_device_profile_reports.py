#!/usr/bin/env python3
"""Generate auditable reports from physical ESP32-CAM profiler logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from vww_esp32.device_profiling import write_device_profile_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser(
        "single", help="Generate one model's device profile artifacts."
    )
    single.add_argument("--serial-log", type=Path, required=True)
    single.add_argument("--model", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "single":
        profile, summary = write_device_profile_artifacts(
            serial_log=args.serial_log,
            model_path=args.model,
            output_dir=args.output_dir,
        )
        print(
            f"Wrote {len(profile)} operator rows to {args.output_dir}; "
            f"Invoke mean={float(summary['mean_us']) / 1000:.3f} ms, "
            f"arena={int(summary['arena_used_bytes']):,} B"
        )


if __name__ == "__main__":
    main()
