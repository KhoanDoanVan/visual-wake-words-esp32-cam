from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def convert_full_integer(model, representative_data, output_path: str | Path) -> Path:
    """Export a Keras model as fully quantized INT8 TFLite for TFLite Micro."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converted = converter.convert()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converted)
    return output_path


def inspect_tflite(model_path: str | Path) -> dict[str, object]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    # XNNPACK inserts a host-only DELEGATE pseudo-op after allocation; it is not
    # serialized in the FlatBuffer and must not be copied into a TFLM resolver.
    ops = sorted(
        {
            item["op_name"]
            for item in interpreter._get_ops_details()
            if item["op_name"] != "DELEGATE"
        }
    )

    def tensor_detail(detail):
        scale, zero_point = detail["quantization"]
        return {
            "name": detail["name"],
            "shape": detail["shape"].tolist(),
            "dtype": np.dtype(detail["dtype"]).name,
            "scale": float(scale),
            "zero_point": int(zero_point),
        }

    return {
        "size_bytes": Path(model_path).stat().st_size,
        "input": tensor_detail(input_detail),
        "output": tensor_detail(output_detail),
        "operators": ops,
    }


def run_tflite(model_path: str | Path, images: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail, output_detail = (
        interpreter.get_input_details()[0],
        interpreter.get_output_details()[0],
    )
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    outputs = []
    for image in images:
        quantized = np.round(image / in_scale + in_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)[None, ...]
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0, 0]
        outputs.append((float(raw) - out_zero) * out_scale)
    return np.asarray(outputs)


def write_c_header(
    model_path: str | Path,
    header_path: str | Path,
    array_name: str = "g_vww_model_data",
) -> Path:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", array_name):
        raise ValueError("array_name must be a valid C identifier")
    data = Path(model_path).read_bytes()
    guard = f"{array_name.upper()}_H_"
    rows = []
    for offset in range(0, len(data), 12):
        rows.append("  " + ", ".join(f"0x{byte:02x}" for byte in data[offset : offset + 12]))
    body = ",\n".join(rows)
    text = (
        f"#ifndef {guard}\n#define {guard}\n\n#include <cstddef>\n#include <cstdint>\n\n"
        f"alignas(16) const unsigned char {array_name}[] = {{\n{body}\n}};\n"
        f"const unsigned int {array_name}_len = {len(data)};\n\n#endif  // {guard}\n"
    )
    header_path = Path(header_path)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(text, encoding="utf-8")
    return header_path
