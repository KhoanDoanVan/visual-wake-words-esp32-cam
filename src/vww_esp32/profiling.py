"""Static, pre-optimization profiling for trained Keras models."""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


def _tensors(value: Any) -> list[Any]:
    """Flatten a Keras tensor or nested tensor structure."""
    if value is None:
        return []
    return list(tf.nest.flatten(value))


def _layer_tensors(layer: tf.keras.layers.Layer, attribute: str) -> list[Any]:
    try:
        return _tensors(getattr(layer, attribute))
    except (AttributeError, ValueError):
        return []


def _shape_tuple(tensor: Any) -> tuple[int | None, ...]:
    shape = getattr(tensor, "shape", ())
    return tuple(int(value) if value is not None else None for value in shape)


def _shape_text(tensors: list[Any]) -> str:
    shapes = [_shape_tuple(tensor) for tensor in tensors]
    if not shapes:
        return "—"
    return str(shapes[0] if len(shapes) == 1 else shapes)


def tensor_elements(tensor: Any, batch_size: int = 1) -> int:
    """Return tensor elements, replacing an unknown batch dimension."""
    shape = _shape_tuple(tensor)
    if not shape:
        return 0
    dimensions = [batch_size if index == 0 and value is None else value for index, value in enumerate(shape)]
    if any(value is None for value in dimensions):
        return 0
    return int(prod(dimensions))


def estimate_layer_macs(layer: tf.keras.layers.Layer, batch_size: int = 1) -> int:
    """Estimate multiply-accumulates for Conv2D, DepthwiseConv2D, and Dense."""
    outputs = _layer_tensors(layer, "output")
    inputs = _layer_tensors(layer, "input")
    if not outputs or not inputs:
        return 0

    output_shape = _shape_tuple(outputs[0])
    input_shape = _shape_tuple(inputs[0])
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D) and len(output_shape) == 4:
        _, output_height, output_width, output_channels = output_shape
        kernel_height, kernel_width = layer.kernel_size
        if None in (output_height, output_width, output_channels):
            return 0
        return int(
            batch_size * output_height * output_width * output_channels * kernel_height * kernel_width
        )
    if isinstance(layer, tf.keras.layers.Conv2D) and len(output_shape) == 4:
        _, output_height, output_width, output_channels = output_shape
        input_channels = input_shape[-1]
        kernel_height, kernel_width = layer.kernel_size
        if None in (output_height, output_width, output_channels, input_channels):
            return 0
        return int(
            batch_size
            * output_height
            * output_width
            * output_channels
            * kernel_height
            * kernel_width
            * input_channels
            / layer.groups
        )
    if isinstance(layer, tf.keras.layers.Dense):
        input_features = input_shape[-1]
        if input_features is None:
            return 0
        return int(batch_size * input_features * layer.units)
    return 0


def _stage(name: str, layer_type: str) -> str:
    if layer_type == "InputLayer":
        return "input"
    if name.startswith("augment") or name == "normalize":
        return "preprocess"
    if name.startswith("stem"):
        return "stem"
    if name.startswith("block"):
        return name.split("_")[0]
    return "head"


def activation_liveness(model: tf.keras.Model, batch_size: int = 1) -> pd.DataFrame:
    """Estimate live activation memory across a Functional Keras graph.

    The estimate keeps a tensor until its final consumer executes. It excludes
    kernel workspaces and allocator reuse, so it is a graph-level comparison
    baseline rather than a device tensor-arena measurement.
    """
    consumers: Counter[int] = Counter()
    for layer in model.layers:
        for tensor in _layer_tensors(layer, "input"):
            consumers[id(tensor)] += 1

    model_output_ids = {id(tensor) for tensor in _tensors(model.outputs)}
    live: dict[int, tuple[int, str]] = {}
    for tensor in _tensors(model.inputs):
        live[id(tensor)] = (tensor_elements(tensor, batch_size), getattr(tensor, "name", "input"))

    trace: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        inputs = _layer_tensors(layer, "input")
        outputs = _layer_tensors(layer, "output")
        for tensor in outputs:
            live[id(tensor)] = (
                tensor_elements(tensor, batch_size),
                getattr(tensor, "name", layer.name),
            )

        live_elements_before_release = sum(elements for elements, _ in live.values())
        trace.append(
            {
                "layer_index": layer_index,
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "live_tensors": len(live),
                "live_elements": live_elements_before_release,
                "live_float32_bytes": live_elements_before_release * 4,
                "live_int8_bytes_hypothetical": live_elements_before_release,
            }
        )

        for tensor in inputs:
            tensor_id = id(tensor)
            consumers[tensor_id] -= 1
            if consumers[tensor_id] <= 0 and tensor_id not in model_output_ids:
                live.pop(tensor_id, None)

    return pd.DataFrame(trace)


def profile_model(
    model: tf.keras.Model,
    batch_size: int = 1,
    training_batch_size: int = 64,
    near_zero_threshold: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create layer, liveness, and model-level pre-optimization profiles."""
    rows: list[dict[str, Any]] = []
    total_weight_elements = 0
    total_weight_zeros = 0
    total_weight_near_zeros = 0
    total_kernel_elements = 0
    total_kernel_zeros = 0
    total_kernel_near_zeros = 0

    for layer_index, layer in enumerate(model.layers):
        inputs = _layer_tensors(layer, "input")
        outputs = _layer_tensors(layer, "output")
        weights = [np.asarray(variable.numpy()) for variable in layer.weights]
        trainable_params = int(sum(prod(variable.shape) for variable in layer.trainable_weights))
        non_trainable_params = int(sum(prod(variable.shape) for variable in layer.non_trainable_weights))
        weight_elements = int(sum(array.size for array in weights))
        weight_bytes = int(sum(array.nbytes for array in weights))
        exact_zeros = int(sum(np.count_nonzero(array == 0) for array in weights))
        near_zeros = int(
            sum(np.count_nonzero(np.abs(array) <= near_zero_threshold) for array in weights)
        )
        kernels = [
            np.asarray(variable.numpy())
            for variable in layer.trainable_weights
            if len(variable.shape) >= 2 and "kernel" in variable.name.lower()
        ]
        kernel_elements = int(sum(array.size for array in kernels))
        kernel_zeros = int(sum(np.count_nonzero(array == 0) for array in kernels))
        kernel_near_zeros = int(
            sum(np.count_nonzero(np.abs(array) <= near_zero_threshold) for array in kernels)
        )
        total_weight_elements += weight_elements
        total_weight_zeros += exact_zeros
        total_weight_near_zeros += near_zeros
        total_kernel_elements += kernel_elements
        total_kernel_zeros += kernel_zeros
        total_kernel_near_zeros += kernel_near_zeros
        output_elements = int(sum(tensor_elements(tensor, batch_size) for tensor in outputs))
        output_shape = _shape_tuple(outputs[0]) if len(outputs) == 1 else ()

        rows.append(
            {
                "layer_index": layer_index,
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "stage": _stage(layer.name, layer.__class__.__name__),
                "input_shape": _shape_text(inputs),
                "output_shape": _shape_text(outputs),
                "output_height": output_shape[1] if len(output_shape) == 4 else np.nan,
                "output_width": output_shape[2] if len(output_shape) == 4 else np.nan,
                "output_channels": output_shape[-1] if len(output_shape) >= 2 else np.nan,
                "parameters": int(layer.count_params()),
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "weight_bytes": weight_bytes,
                "weight_sparsity": exact_zeros / weight_elements if weight_elements else np.nan,
                "near_zero_fraction": near_zeros / weight_elements if weight_elements else np.nan,
                "kernel_sparsity": kernel_zeros / kernel_elements if kernel_elements else np.nan,
                "output_elements_batch1": output_elements,
                "output_float32_bytes_batch1": output_elements * 4,
                "output_int8_bytes_batch1_hypothetical": output_elements,
                "estimated_macs_batch1": estimate_layer_macs(layer, batch_size),
            }
        )

    layers = pd.DataFrame(rows)
    liveness = activation_liveness(model, batch_size=batch_size)
    total_parameters = int(model.count_params())
    total_trainable = int(sum(prod(variable.shape) for variable in model.trainable_weights))
    total_non_trainable = int(sum(prod(variable.shape) for variable in model.non_trainable_weights))
    total_weight_bytes = int(sum(np.asarray(variable.numpy()).nbytes for variable in model.weights))
    peak_index = int(liveness.live_float32_bytes.idxmax()) if len(liveness) else 0
    peak_row = liveness.loc[peak_index] if len(liveness) else None
    training_activation_bytes = int(
        layers.output_elements_batch1.sum() * 4 * training_batch_size
    )
    trainable_bytes = total_trainable * 4

    summary = {
        "model_name": model.name,
        "layers": len(model.layers),
        "parameters": total_parameters,
        "trainable_parameters": total_trainable,
        "non_trainable_parameters": total_non_trainable,
        "float32_weight_bytes": total_weight_bytes,
        "global_weight_sparsity": float(total_weight_zeros / total_weight_elements)
        if total_weight_elements
        else 0.0,
        "global_weight_near_zero_fraction": float(
            total_weight_near_zeros / total_weight_elements
        )
        if total_weight_elements
        else 0.0,
        "global_kernel_sparsity": float(total_kernel_zeros / total_kernel_elements)
        if total_kernel_elements
        else 0.0,
        "global_kernel_near_zero_fraction": float(
            total_kernel_near_zeros / total_kernel_elements
        )
        if total_kernel_elements
        else 0.0,
        "estimated_macs_batch1": int(layers.estimated_macs_batch1.sum()),
        "peak_live_activation_float32_bytes_batch1": int(
            liveness.live_float32_bytes.max() if len(liveness) else 0
        ),
        "peak_live_activation_int8_bytes_batch1_hypothetical": int(
            liveness.live_int8_bytes_hypothetical.max() if len(liveness) else 0
        ),
        "peak_live_activation_layer": str(peak_row.layer_name) if peak_row is not None else None,
        "float32_inference_structural_bytes": int(
            total_weight_bytes
            + (liveness.live_float32_bytes.max() if len(liveness) else 0)
        ),
        "training_batch_size_for_estimate": int(training_batch_size),
        "training_structural_bytes_estimate": int(
            total_weight_bytes
            + trainable_bytes  # gradients
            + 2 * trainable_bytes  # Adam first/second moment slots
            + training_activation_bytes
        ),
        "near_zero_threshold": float(near_zero_threshold),
        "memory_estimate_scope": (
            "Static graph estimate, not an ESP32 SRAM or tensor-arena measurement. Weights and "
            "activations may occupy different memory regions; fusion, in-place reuse, kernel "
            "workspace, allocator fragmentation, and runtime overhead are not modeled."
        ),
    }

    for column in ("parameters", "estimated_macs_batch1", "output_float32_bytes_batch1"):
        total = float(layers[column].sum())
        layers[f"{column}_share"] = layers[column] / total if total else 0.0

    return layers, liveness, summary


def human_bytes(value: float) -> str:
    """Format a byte count with binary units."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(amount) < 1024 or unit == "GiB":
            return f"{amount:,.2f} {unit}"
        amount /= 1024
    return f"{amount:,.2f} GiB"
