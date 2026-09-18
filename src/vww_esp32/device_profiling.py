"""Parse and enrich real ESP32 TFLite Micro operator profiles."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

PROFILE_MARKER = "VWW_PROFILE,"
TENSOR_ARENA_RESERVED_BYTES = 350 * 1024
DASHBOARD_JPEG_BUFFER_BYTES = 184_320
STREAM_CLIENT_BUFFER_BYTES = 184_320


def _parse_value(value: str) -> Any:
    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            pass
    return value


def parse_device_profile_text(text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one complete VWW_PROFILE block from an ESP32 serial log."""
    metadata: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    saw_end = False

    for raw_line in text.splitlines():
        marker_at = raw_line.find(PROFILE_MARKER)
        if marker_at < 0:
            continue
        profile_line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line[marker_at:]).strip()
        fields = profile_line.split(",")
        record_type = fields[1]
        values = {
            key: _parse_value(value)
            for field in fields[2:]
            if "=" in field
            for key, value in [field.split("=", 1)]
        }
        if record_type == "BEGIN":
            metadata = values
        elif record_type == "OP":
            rows.append(values)
        elif record_type == "TOTAL":
            summary = values
        elif record_type == "END":
            saw_end = True

    if not metadata or not rows or not summary or not saw_end:
        raise ValueError("serial log does not contain one complete VWW_PROFILE block")
    operators = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    expected = list(range(len(operators)))
    if operators["index"].tolist() != expected:
        raise ValueError("operator indexes in the device profile are incomplete or unordered")
    profile_summary = {**metadata, **summary, "operators": len(operators)}
    return operators, profile_summary


def parse_device_profile(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    return parse_device_profile_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def _tensor_bytes(detail: dict[str, Any]) -> int:
    shape = [int(value) for value in detail["shape"]]
    elements = prod(shape) if shape else 1
    return int(elements * np.dtype(detail["dtype"]).itemsize)


def _shape_text(detail: dict[str, Any] | None) -> str:
    if detail is None:
        return "—"
    return "x".join(str(int(value)) for value in detail["shape"])


def tflite_operator_inventory(model_path: str | Path) -> pd.DataFrame:
    """Return fused TFLite operator shapes, storage, and batch-1 compute estimates."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    tensor_details = {detail["index"]: detail for detail in interpreter.get_tensor_details()}
    graph_inputs = {int(detail["index"]) for detail in interpreter.get_input_details()}
    graph_outputs = {int(detail["index"]) for detail in interpreter.get_output_details()}
    ops = [op for op in interpreter._get_ops_details() if op["op_name"] != "DELEGATE"]
    produced = {int(index) for op in ops for index in op["outputs"] if int(index) >= 0}
    producer = {
        int(tensor_index): operator_index
        for operator_index, op in enumerate(ops)
        for tensor_index in op["outputs"]
        if int(tensor_index) >= 0
    }
    consumers: dict[int, list[int]] = defaultdict(list)
    for operator_index, op in enumerate(ops):
        for tensor_index in op["inputs"]:
            if int(tensor_index) >= 0:
                consumers[int(tensor_index)].append(operator_index)
    activation_indexes = graph_inputs | produced
    activation_bytes = {
        index: _tensor_bytes(tensor_details[index])
        for index in activation_indexes
        if index in tensor_details
    }
    last_consumer = {
        index: max(consumers.get(index, [-1])) for index in activation_indexes
    }
    for index in graph_outputs:
        last_consumer[index] = len(ops)
    rows: list[dict[str, Any]] = []

    for operator_index, op in enumerate(ops):
        inputs = [int(index) for index in op["inputs"] if int(index) >= 0]
        outputs = [int(index) for index in op["outputs"] if int(index) >= 0]
        input_detail = tensor_details.get(inputs[0]) if inputs else None
        output_detail = tensor_details.get(outputs[0]) if outputs else None
        constant_indexes = [
            index for index in inputs if index not in produced and index not in graph_inputs
        ]
        activation_input_indexes = [index for index in inputs if index in activation_indexes]
        constant_bytes = sum(_tensor_bytes(tensor_details[index]) for index in constant_indexes)
        live_before = {
            index
            for index in activation_indexes
            if producer.get(index, -1) < operator_index
            and last_consumer.get(index, -1) >= operator_index
        }
        live_during = live_before | {index for index in outputs if index in activation_indexes}
        live_after = {
            index
            for index in activation_indexes
            if producer.get(index, -1) <= operator_index
            and (
                last_consumer.get(index, -1) > operator_index
                or index in graph_outputs
            )
        }
        live_before_bytes = sum(activation_bytes.get(index, 0) for index in live_before)
        live_during_bytes = sum(activation_bytes.get(index, 0) for index in live_during)
        live_after_bytes = sum(activation_bytes.get(index, 0) for index in live_after)
        all_input_activation_bytes = sum(
            activation_bytes.get(index, 0) for index in activation_input_indexes
        )
        all_output_activation_bytes = sum(
            activation_bytes.get(index, 0) for index in outputs
        )
        output_shape = tuple(int(value) for value in output_detail["shape"]) if output_detail else ()
        weight_detail = tensor_details.get(inputs[1]) if len(inputs) > 1 else None
        weight_shape = tuple(int(value) for value in weight_detail["shape"]) if weight_detail else ()
        macs = 0
        elementwise_ops = 0
        if op["op_name"] == "CONV_2D" and len(output_shape) == 4 and len(weight_shape) == 4:
            _, height, width, output_channels = output_shape
            _, kernel_height, kernel_width, input_channels = weight_shape
            macs = height * width * output_channels * kernel_height * kernel_width * input_channels
        elif (
            op["op_name"] == "DEPTHWISE_CONV_2D"
            and len(output_shape) == 4
            and len(weight_shape) == 4
        ):
            _, height, width, output_channels = output_shape
            _, kernel_height, kernel_width, _ = weight_shape
            macs = height * width * output_channels * kernel_height * kernel_width
        elif op["op_name"] == "FULLY_CONNECTED" and len(weight_shape) == 2:
            macs = weight_shape[0] * weight_shape[1]
        elif op["op_name"] == "MEAN" and input_detail is not None:
            elementwise_ops = _tensor_bytes(input_detail)
        elif output_detail is not None:
            elementwise_ops = int(prod(output_shape))

        rows.append(
            {
                "index": operator_index,
                "operator": op["op_name"],
                "input_shape": _shape_text(input_detail),
                "output_shape": _shape_text(output_detail),
                "input_activation_bytes": all_input_activation_bytes,
                "output_activation_bytes": all_output_activation_bytes,
                "activation_io_bytes": all_input_activation_bytes
                + all_output_activation_bytes,
                "constant_bytes": constant_bytes,
                "live_activation_before_bytes": live_before_bytes,
                "live_activation_during_bytes": live_during_bytes,
                "live_activation_after_bytes": live_after_bytes,
                "live_tensor_count_during": len(live_during),
                "live_plus_constants_bytes": live_during_bytes + constant_bytes,
                "estimated_macs": int(macs),
                "estimated_elementwise_ops": int(elementwise_ops),
            }
        )
    return pd.DataFrame(rows)


def enrich_device_profile(
    measured: pd.DataFrame, model_path: str | Path
) -> pd.DataFrame:
    """Join board timings to the same-index operators in a TFLite FlatBuffer."""
    inventory = tflite_operator_inventory(model_path)
    if len(measured) != len(inventory):
        raise ValueError(
            f"device reported {len(measured)} operators but model contains {len(inventory)}"
        )
    profile = inventory.merge(measured, on="index", how="inner", validate="one_to_one")
    tag_mismatch = profile[profile["tag"] != profile["operator"]]
    if len(tag_mismatch):
        raise ValueError(
            "device/model operator mismatch: "
            + tag_mismatch[["index", "tag", "operator"]].to_dict("records").__repr__()
        )
    profile["mean_ms"] = profile["mean_us"] / 1000
    profile["min_ms"] = profile["min_us"] / 1000
    profile["max_ms"] = profile["max_us"] / 1000
    profile["latency_share"] = profile["mean_us"] / profile["mean_us"].sum()
    profile["mmacs_per_second"] = np.where(
        profile["estimated_macs"] > 0,
        profile["estimated_macs"] / profile["mean_us"],
        np.nan,
    )
    profile["activation_kib_per_ms"] = np.where(
        profile["mean_ms"] > 0,
        profile["activation_io_bytes"] / 1024 / profile["mean_ms"],
        np.nan,
    )
    return profile


def _runtime_memory_from_serial(text: str) -> dict[str, int | float]:
    """Extract memory, camera, bandwidth, and pipeline timings from a serial log."""
    result: dict[str, int | float] = {}
    snapshot_pattern = re.compile(
        r"(?P<stage>before allocations|ready): internal_free=(?P<internal_free>\d+) "
        r"internal_largest=(?P<internal_largest>\d+) psram_free=(?P<psram_free>\d+) "
        r"psram_largest=(?P<psram_largest>\d+)"
    )
    for match in snapshot_pattern.finditer(text):
        prefix = "before" if match.group("stage") == "before allocations" else "ready"
        for field in ("internal_free", "internal_largest", "psram_free", "psram_largest"):
            result[f"{prefix}_{field}_bytes"] = int(match.group(field))
    camera_match = re.search(r"Allocating (\d+) Byte frame buffer in PSRAM", text)
    if camera_match:
        result["camera_framebuffer_bytes"] = int(camera_match.group(1))
    capture_match = re.search(r"Inference capture size set to (\d+)x(\d+)", text)
    if capture_match:
        result["capture_width"] = int(capture_match.group(1))
        result["capture_height"] = int(capture_match.group(2))
        result["rgb888_scratch_bytes"] = (
            result["capture_width"] * result["capture_height"] * 3
        )
    for label, key in (
        ("internal SRAM", "internal_sram_memcpy_mib_s"),
        ("mapped PSRAM", "mapped_psram_memcpy_mib_s"),
    ):
        match = re.search(rf"{label} memcpy: ([0-9.]+) MiB/s", text)
        if match:
            result[key] = float(match.group(1))
    camera_benchmark = re.search(
        r"Camera-only: ([0-9.]+) fps .*?JPEG mean=([0-9.]+)B .*?payload=([0-9.]+)kbit/s",
        text,
    )
    if camera_benchmark:
        result["camera_only_fps"] = float(camera_benchmark.group(1))
        result["camera_jpeg_mean_bytes"] = float(camera_benchmark.group(2))
        result["camera_payload_kbit_s"] = float(camera_benchmark.group(3))
    timing_pattern = re.compile(
        r"timing_ms\[capture=([0-9.]+) preprocess=([0-9.]+) infer=([0-9.]+) "
        r"publish=([0-9.]+) active=([0-9.]+) period=([0-9.]+) fps=([0-9.]+)\]"
    )
    timing_rows = np.asarray(
        [[float(value) for value in match] for match in timing_pattern.findall(text)],
        dtype=np.float64,
    )
    if timing_rows.size:
        names = ("capture", "preprocess", "infer", "publish", "active", "period", "fps")
        for index, name in enumerate(names):
            result[f"pipeline_{name}_median_ms" if name != "fps" else "pipeline_fps_median"] = float(
                np.median(timing_rows[:, index])
            )
    return result


def write_device_profile_artifacts(
    serial_log: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create CSV, JSON, and Markdown artifacts from one physical-board run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serial_text = Path(serial_log).read_text(encoding="utf-8", errors="replace")
    measured, summary = parse_device_profile_text(serial_text)
    profile = enrich_device_profile(measured, model_path)
    runtime_memory = _runtime_memory_from_serial(serial_text)
    live_peak_row = profile.loc[profile["live_activation_during_bytes"].idxmax()]
    arena_used = int(summary["arena_used_bytes"])
    live_peak = int(live_peak_row["live_activation_during_bytes"])
    input_tensor_bytes = int(profile.iloc[0]["input_activation_bytes"])
    summary = {
        **summary,
        **runtime_memory,
        "model_path": str(Path(model_path)),
        "model_bytes": Path(model_path).stat().st_size,
        "estimated_macs": int(profile["estimated_macs"].sum()),
        "operator_profile_mean_us": float(profile["mean_us"].sum()),
        "tflite_live_activation_peak_bytes": live_peak,
        "tflite_live_activation_peak_operator_index": int(live_peak_row["index"]),
        "tflite_live_activation_peak_operator": str(live_peak_row["operator"]),
        "model_constant_bytes": int(profile["constant_bytes"].sum()),
        "input_tensor_bytes": input_tensor_bytes,
        "tensor_arena_reserved_bytes": TENSOR_ARENA_RESERVED_BYTES,
        "tensor_arena_headroom_bytes": TENSOR_ARENA_RESERVED_BYTES - arena_used,
        "tensor_arena_utilization": arena_used / TENSOR_ARENA_RESERVED_BYTES,
        "arena_bytes_beyond_live_tensors": arena_used - live_peak,
        "dashboard_jpeg_buffer_bytes": DASHBOARD_JPEG_BUFFER_BYTES,
        "stream_client_buffer_bytes": STREAM_CLIENT_BUFFER_BYTES,
    }
    if "before_psram_free_bytes" in summary and "ready_psram_free_bytes" in summary:
        summary["psram_consumed_before_to_ready_bytes"] = (
            summary["before_psram_free_bytes"] - summary["ready_psram_free_bytes"]
        )
    if "before_internal_free_bytes" in summary and "ready_internal_free_bytes" in summary:
        summary["internal_consumed_before_to_ready_bytes"] = (
            summary["before_internal_free_bytes"] - summary["ready_internal_free_bytes"]
        )
    profile.to_csv(output_dir / "operator_profile.csv", index=False)
    (output_dir / "device_profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    memory_rows = [
        ("Internal heap free at ready", summary.get("ready_internal_free_bytes"), "Internal SRAM", "measured"),
        ("Largest internal block at ready", summary.get("ready_internal_largest_bytes"), "Internal SRAM", "measured"),
        ("Mapped PSRAM free at ready", summary.get("ready_psram_free_bytes"), "PSRAM", "measured"),
        ("PSRAM consumed from pre-allocation to ready", summary.get("psram_consumed_before_to_ready_bytes"), "PSRAM", "measured delta"),
        ("Tensor arena reservation", summary["tensor_arena_reserved_bytes"], "PSRAM", "configured allocation"),
        ("Tensor arena used", arena_used, "PSRAM arena", "measured by TFLM"),
        ("TFLite live activation peak", live_peak, "PSRAM arena", "graph-liveness estimate"),
        ("Arena bytes beyond live tensors", summary["arena_bytes_beyond_live_tensors"], "PSRAM arena", "derived"),
        ("RGB888 preprocessing scratch", summary.get("rgb888_scratch_bytes"), "PSRAM", "configured allocation"),
        ("Camera framebuffer", summary.get("camera_framebuffer_bytes"), "PSRAM", "driver-reported allocation"),
        ("Dashboard JPEG buffer", summary["dashboard_jpeg_buffer_bytes"], "PSRAM", "configured allocation"),
        ("Additional connected stream-client buffer", summary["stream_client_buffer_bytes"], "PSRAM", "conditional allocation"),
        ("Model constants referenced by operators", summary["model_constant_bytes"], "Flash", "FlatBuffer inventory"),
        ("Complete TFLite FlatBuffer", summary["model_bytes"], "Flash", "file size"),
    ]
    memory_frame = pd.DataFrame(
        [
            {"item": item, "bytes": int(value), "region": region, "basis": basis}
            for item, value, region, basis in memory_rows
            if value is not None
        ]
    )
    memory_frame.to_csv(output_dir / "memory_summary.csv", index=False)

    top = profile.nlargest(10, "mean_us")
    top_rows = "\n".join(
        f"| {int(row['index'])} | {row['operator']} | {row['input_shape']} | "
        f"{row['output_shape']} | {int(row['estimated_macs']):,} | "
        f"{row['mean_ms']:.3f} | {row['latency_share']:.1%} |"
        for _, row in top.iterrows()
    )
    top_memory = profile.nlargest(10, "live_activation_during_bytes")
    memory_operator_rows = "\n".join(
        f"| {int(row['index'])} | {row['operator']} | {row['input_shape']} | "
        f"{row['output_shape']} | {int(row['activation_io_bytes']):,} | "
        f"{int(row['live_activation_during_bytes']):,} | {int(row['constant_bytes']):,} |"
        for _, row in top_memory.iterrows()
    )
    report = f"""# ESP32-CAM device operator profile: {summary['variant']}

This report contains timings measured inside TFLite Micro on the physical ESP32-CAM.
It uses batch 1, {summary['samples']} measured invocations after {summary['warmup']} warm-up
invocations, and input `{summary['input']}`.

> These timings include lightweight per-operator profiler hooks. Use them for operator
> attribution and controlled comparisons between identically instrumented builds. The
> uninstrumented production firmware remains the source of truth for final user-facing
> latency and FPS.

## Summary

| Metric | Value |
|---|---:|
| Full Invoke mean | {float(summary['mean_us']) / 1000:.3f} ms |
| Full Invoke range | {float(summary['min_us']) / 1000:.3f}-{float(summary['max_us']) / 1000:.3f} ms |
| Sum of profiled operators | {float(summary['operator_sum_mean_us']) / 1000:.3f} ms |
| Dispatch/profiler remainder | {float(summary['unprofiled_mean_us']) / 1000:.3f} ms |
| Tensor arena actually used | {int(summary['arena_used_bytes']):,} B |
| Tensor arena reservation | {int(summary['tensor_arena_reserved_bytes']):,} B |
| Tensor arena utilization | {float(summary['tensor_arena_utilization']):.1%} |
| TFLite live activation peak | {int(summary['tflite_live_activation_peak_bytes']):,} B at operator {int(summary['tflite_live_activation_peak_operator_index'])} |
| Arena bytes beyond live-tensor estimate | {int(summary['arena_bytes_beyond_live_tensors']):,} B |
| Internal SRAM free / largest block at ready | {int(summary['ready_internal_free_bytes']):,} / {int(summary['ready_internal_largest_bytes']):,} B |
| Mapped PSRAM free at ready | {int(summary['ready_psram_free_bytes']):,} B |
| TFLite model | {int(summary['model_bytes']):,} B |
| Operator constants in FlatBuffer | {int(summary['model_constant_bytes']):,} B |
| Estimated MACs | {int(summary['estimated_macs']):,} |
| Fused TFLite operators | {int(summary['operators'])} |

## Ten slowest operators

| Index | Operator | Input | Output | MACs | Mean ms | Latency share |
|---:|---|---|---|---:|---:|---:|
{top_rows}

## Operators with the largest activation footprint

| Index | Operator | Input | Output | Activation I/O | Live during op | Constants |
|---:|---|---|---|---:|---:|---:|
{memory_operator_rows}

## Device memory accounting

`memory_summary.csv` separates measured heap values, configured allocations, and static
FlatBuffer estimates. Important nesting rules apply: arena-used bytes are inside the arena
reservation, and live activation bytes are inside arena-used bytes. Model constants remain
in flash and must not be added to SRAM or PSRAM use. The stream-client buffer is allocated
only while a browser consumes the MJPEG endpoint.

The theoretical live-activation peak is {live_peak:,} B. TFLite Micro reports
{arena_used:,} B used in its arena, leaving {int(summary['tensor_arena_headroom_bytes']):,} B
unused inside the conservative {int(summary['tensor_arena_reserved_bytes']):,} B reservation.
The difference between arena-used and live-tensor estimates includes persistent tensors,
operator scratch buffers, allocator metadata, alignment, and reuse decisions that cannot be
assigned reliably to a single operator from runtime timing hooks.

## Interpretation

Operator indexes refer to the deployed TFLite FlatBuffer, not unfused Keras layer indexes.
Batch normalization and ReLU are folded into quantized convolution operators during export.
`operator_profile.csv` contains every operator, tensor shape, activation I/O, live-tensor
memory before/during/after execution, constant bytes, MAC estimate, minimum/mean/maximum
device latency, latency share, and effective compute and activation throughput.
"""
    (output_dir / "DEVICE_LAYER_PROFILE_REPORT.md").write_text(report, encoding="utf-8")
    return profile, summary


def load_device_profile(output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir = Path(output_dir)
    profile = pd.read_csv(output_dir / "operator_profile.csv")
    summary = json.loads((output_dir / "device_profile_summary.json").read_text())
    return profile, summary
