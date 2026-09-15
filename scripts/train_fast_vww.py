#!/usr/bin/env python3
"""Fine-tune, evaluate, and fully quantize a lower-resolution VWW candidate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from vww_esp32.config import load_config, resolve_paths, seed_everything
from vww_esp32.evaluation import (
    classification_metrics,
    collect_predictions,
    expected_calibration_error,
    select_threshold,
)
from vww_esp32.exporting import (
    convert_full_integer,
    inspect_tflite,
    run_tflite,
    write_c_header,
)
from vww_esp32.modeling import build_tiny_mobilenet_v1, compile_model, training_callbacks
from vww_esp32.preprocessing import make_dataset, representative_dataset, split_manifest
from vww_esp32.profiling import profile_model


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_initial_weights(model, root: Path, checkpoint: str | None) -> str:
    if not checkpoint:
        return "random"
    import tensorflow as tf

    source_path = root / checkpoint
    source = tf.keras.models.load_model(source_path)
    source_weights = source.get_weights()
    target_weights = model.get_weights()
    if [item.shape for item in source_weights] != [item.shape for item in target_weights]:
        raise ValueError(f"Weight shapes do not match initialization checkpoint {source_path}")
    model.set_weights(source_weights)
    return str(source_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast_80.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    # Keep framework caches inside the writable project/temp roots.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vww-matplotlib")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

    config, root = load_config(args.config)
    paths = resolve_paths(config, root)
    reports = paths["artifacts"] / "reports"
    checkpoints = paths["artifacts"] / "checkpoints"
    models = paths["artifacts"] / "models"
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    image_size = tuple(config["preprocessing"]["image_size"])
    batch_size = int(config["preprocessing"]["batch_size"])

    manifest = pd.read_csv(paths["processed"] / "manifest.csv")
    splits = split_manifest(manifest)
    train_ds = make_dataset(
        splits["train"], image_size, batch_size, training=True, seed=seed,
        cache=config["preprocessing"].get("cache", False),
    )
    val_ds = make_dataset(splits["val"], image_size, batch_size, training=False)
    test_ds = make_dataset(splits["test"], image_size, batch_size, training=False)

    model = build_tiny_mobilenet_v1(
        input_shape=(*image_size, int(config["preprocessing"]["channels"])),
        alpha=float(config["model"]["width_multiplier"]),
        dropout=float(config["model"]["dropout"]),
        l2=float(config["model"]["l2"]),
        camera_augmentation=bool(config["model"].get("camera_augmentation", False)),
        camera_augmentation_strength=float(
            config["model"].get("camera_augmentation_strength", 1.0)
        ),
        augmentation_seed=seed,
    )
    initialization = load_initial_weights(
        model, root, config["model"].get("initialization_checkpoint")
    )
    if config["model"].get("freeze_batch_norm", False):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    compile_model(
        model,
        learning_rate=float(config["training"]["learning_rate"]),
        label_smoothing=float(config["training"]["label_smoothing"]),
    )

    _, _, pre_profile = profile_model(model, training_batch_size=batch_size)
    write_json(reports / "pre_training_profile.json", pre_profile)
    print(
        f"candidate input={image_size} params={model.count_params():,} "
        f"macs={pre_profile['estimated_macs_batch1']:,} init={initialization}",
        flush=True,
    )

    # Resolution changes do not alter convolution weight shapes. Preserve a
    # validation-scored copy of the inherited weights so augmentation fine-
    # tuning can never silently replace them with a worse epoch.
    initial_path = checkpoints / "initial.keras"
    model.save(initial_path)
    initial_y_val, initial_p_val = collect_predictions(model, val_ds)
    initial_val_pr_auc = float(average_precision_score(initial_y_val, initial_p_val))

    epochs = args.epochs or int(config["training"]["epochs"])
    started = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks(
            paths["artifacts"],
            monitor=config["training"]["monitor"],
            patience=int(config["training"]["early_stopping_patience"]),
            lr_patience=int(config["training"]["reduce_lr_patience"]),
        ),
    )
    training_seconds = time.monotonic() - started
    best_path = checkpoints / "best.keras"
    trained_model = tf.keras.models.load_model(best_path)
    trained_y_val, trained_p_val = collect_predictions(trained_model, val_ds)
    trained_val_pr_auc = float(average_precision_score(trained_y_val, trained_p_val))
    if initial_val_pr_auc >= trained_val_pr_auc:
        training_model = tf.keras.models.load_model(initial_path)
        chosen_checkpoint = str(initial_path)
    else:
        training_model = trained_model
        chosen_checkpoint = str(best_path)

    # Training-only seeded augmentation layers contain RNG state variables.
    # Copy learned weights into a stateless inference graph so the FlatBuffer
    # contains only operators supported by TFLite Micro.
    model = build_tiny_mobilenet_v1(
        input_shape=(*image_size, int(config["preprocessing"]["channels"])),
        alpha=float(config["model"]["width_multiplier"]),
        dropout=float(config["model"]["dropout"]),
        l2=float(config["model"]["l2"]),
        camera_augmentation=False,
        augmentation_seed=None,
    )
    model.set_weights(training_model.get_weights())
    model.save(models / "final.keras")

    y_val, p_val = collect_predictions(model, val_ds)
    threshold_choice = select_threshold(
        y_val,
        p_val,
        objective=config["evaluation"]["threshold_objective"],
        minimum_recall=float(config["evaluation"]["minimum_recall"]),
        false_positive_cost=float(config["evaluation"]["false_positive_cost"]),
        false_negative_cost=float(config["evaluation"]["false_negative_cost"]),
    )
    threshold = float(threshold_choice["threshold"])
    val_metrics = classification_metrics(y_val, p_val, threshold)
    val_metrics["expected_calibration_error_10_bins"] = expected_calibration_error(y_val, p_val)
    write_json(reports / "validation_metrics.json", val_metrics)

    # The threshold is now locked; touch the held-out test split only once.
    y_test, p_test = collect_predictions(model, test_ds)
    float_metrics = classification_metrics(y_test, p_test, threshold)
    float_metrics["expected_calibration_error_10_bins"] = expected_calibration_error(
        y_test, p_test
    )
    write_json(reports / "test_metrics_float.json", float_metrics)

    model_path = models / config["export"]["model_filename"]
    representative = representative_dataset(
        splits["train"],
        image_size,
        int(config["export"]["representative_samples"]),
        seed,
    )
    convert_full_integer(model, representative, model_path)
    export_info = inspect_tflite(model_path)
    write_json(reports / "export_info.json", export_info)

    test_images = np.concatenate([np.asarray(images) for images, _ in test_ds], axis=0)
    int8_probabilities = run_tflite(model_path, test_images)
    int8_metrics = classification_metrics(y_test, int8_probabilities, threshold)
    int8_metrics["expected_calibration_error_10_bins"] = expected_calibration_error(
        y_test, int8_probabilities
    )
    int8_metrics["mean_absolute_probability_delta_vs_float"] = float(
        np.mean(np.abs(int8_probabilities - p_test))
    )
    write_json(reports / "test_metrics_int8.json", int8_metrics)

    layers, liveness, final_profile = profile_model(model, training_batch_size=batch_size)
    layers.to_csv(reports / "layer_profile.csv", index=False)
    liveness.to_csv(reports / "activation_liveness.csv", index=False)
    final_profile.update(
        {
            "checkpoint": str(best_path),
            "checkpoint_bytes": best_path.stat().st_size,
            "tflite_bytes": model_path.stat().st_size,
        }
    )
    write_json(reports / "model_profile_summary.json", final_profile)

    baseline_metrics = json.loads((root / "artifacts/reports/test_metrics.json").read_text())
    gate_config = config["deployment_gate"]
    tflm_operators = {
        "ADD",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "LOGISTIC",
        "MEAN",
        "MUL",
        "RESHAPE",
    }
    gates = {
        "full_int8_io": export_info["input"]["dtype"] == export_info["output"]["dtype"] == "int8",
        "tflm_operator_set": set(export_info["operators"]) <= tflm_operators,
        "mac_budget": final_profile["estimated_macs_batch1"] <= int(gate_config["max_macs"]),
        "model_size_budget": export_info["size_bytes"] <= int(gate_config["max_model_bytes"]),
        "minimum_test_recall": int8_metrics["recall"] >= float(gate_config["minimum_test_recall"]),
        "minimum_test_f1": int8_metrics["f1"] >= float(gate_config["minimum_test_f1"]),
        "baseline_f1_retained": int8_metrics["f1"] >= (
            float(baseline_metrics["f1"]) - float(gate_config["maximum_f1_drop_from_baseline"])
        ),
    }
    summary = {
        "architecture_basis": "VWW paper MobileNetV1 depthwise-separable chain",
        "initialization": initialization,
        "initial_validation_pr_auc": initial_val_pr_auc,
        "trained_validation_pr_auc": trained_val_pr_auc,
        "chosen_checkpoint": chosen_checkpoint,
        "epochs_completed": len(history.history["loss"]),
        "training_seconds": training_seconds,
        "best_val_pr_auc": max(history.history["val_pr_auc"]),
        "selected_threshold": threshold,
        "baseline_test_metrics": baseline_metrics,
        "validation_metrics": val_metrics,
        "float_test_metrics": float_metrics,
        "int8_test_metrics": int8_metrics,
        "profile": final_profile,
        "export": export_info,
        "deployment_gates": gates,
        "ready_for_firmware": all(gates.values()),
    }
    if summary["ready_for_firmware"]:
        firmware_header = (
            root
            / "firmware"
            / "esp32_cam_vww"
            / "include"
            / config["export"]["header_filename"]
        )
        write_c_header(model_path, firmware_header, config["export"]["c_array_name"])
        summary["firmware_header"] = str(firmware_header)
    write_json(reports / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
