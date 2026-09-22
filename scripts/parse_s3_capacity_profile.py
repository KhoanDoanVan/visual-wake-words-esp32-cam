#!/usr/bin/env python3
"""Parse the ESP32-S3 capacity probe's machine-readable serial records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def fields(record: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in record.split(",")[2:]:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def number(value: str) -> int | float | str:
    try:
        if any(character in value for character in ".eE"):
            return float(value)
        return int(value, 0)
    except ValueError:
        return value


def converted(values: dict[str, str]) -> dict[str, int | float | str]:
    return {key: number(value) for key, value in values.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--firmware", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "source_log": args.log.name,
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "sha256": sha256(args.model),
        },
        "firmware": {
            "path": str(args.firmware),
            "bytes": args.firmware.stat().st_size,
            "sha256": sha256(args.firmware),
        },
        "chip": {},
        "storage": {},
        "models": {},
        "latency": {},
        "profile_totals": {},
    }
    memories: list[dict[str, object]] = []
    bandwidth: list[dict[str, object]] = []
    operators: list[dict[str, object]] = []
    placement = ""

    for raw in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw)
        marker = line.find("S3_CAPACITY,")
        profile_marker = line.find("VWW_PROFILE,")
        if marker >= 0:
            record = line[marker:]
            kind = record.split(",", 2)[1]
            values = converted(fields(record))
            if kind == "CHIP":
                summary["chip"] = values
            elif kind == "STORAGE":
                summary["storage"] = values
            elif kind == "MEM":
                memories.append(values)
            elif kind == "BANDWIDTH":
                bandwidth.append(values)
            elif kind == "MODEL" and values.get("status") == "ready":
                summary["models"][str(values["placement"])] = values
            elif kind == "LATENCY":
                summary["latency"][str(values["placement"])] = values
        elif profile_marker >= 0:
            record = line[profile_marker:]
            kind = record.split(",", 2)[1]
            values = converted(fields(record))
            if kind == "BEGIN":
                variant = str(values.get("variant", ""))
                placement = variant.rsplit("_", 1)[-1]
            elif kind == "OP":
                operators.append({"placement": placement, **values})
            elif kind == "TOTAL":
                summary["profile_totals"][placement] = values

    summary["derived"] = {}
    latency = summary["latency"]
    if "internal" in latency and "psram" in latency:
        internal = float(latency["internal"]["mean_us"])
        psram = float(latency["psram"]["mean_us"])
        summary["derived"] = {
            "internal_model_fps": 1_000_000.0 / internal,
            "psram_model_fps": 1_000_000.0 / psram,
            "internal_vs_psram_speedup": psram / internal,
            "internal_latency_reduction_percent": 100.0 * (psram - internal) / psram,
            "internal_effective_mmac_per_s": 3.993536 / (internal / 1_000_000.0),
            "psram_effective_mmac_per_s": 3.993536 / (psram / 1_000_000.0),
        }

    (args.output_dir / "device_profile_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "memory_profile.csv", memories)
    write_csv(args.output_dir / "bandwidth_profile.csv", bandwidth)
    write_csv(args.output_dir / "operator_profile.csv", operators)


if __name__ == "__main__":
    main()
