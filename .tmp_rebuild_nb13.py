from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "optimization" / "pruning" / "13_unstructured_magnitude_pruning.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        """
        # 13 · Unstructured magnitude pruning: control and paper-inspired recovery

        **Single responsibility:** study fine-grained weight pruning on the frozen Fast-80
        pruning baseline without confusing mathematical sparsity with ESP32 speed.

        This revision deliberately contains two branches:

        1. **Global one-shot control** — exact global `|weight|` ranking followed by two masked
           recovery epochs. This preserves the original negative hardware control.
        2. **Paper-inspired iterative method** — layer sensitivity changes each layer's effective
           magnitude threshold, five prune/retrain rounds preserve surviving weights, and a
           validation-controlled final recovery uses one-tenth of Fast-80's training rate.

        The second branch is inspired by Han et al., not presented as an exact reproduction of
        their AlexNet/VGG experiments. Fast-80 has depthwise-separable layers, BatchNorm, a binary
        head, and a different dataset.
        """
    ),
    md(
        """
        ## Evidence flow

        ```text
        Frozen Fast-80 + frozen train/validation/test split
                         │
                ┌────────┴────────┐
                │                 │
        global one-shot      layer-sensitivity audit
                │                 │
          2-epoch recovery   5 × prune/retrain + early stop
                └────────┬────────┘
                         ↓
        validation-only float and INT8 gates
                         ↓
              one selected deployable model
                         ↓
        held-out test, storage, MACs, activation, host timing
                         ↓
        exact FlatBuffer physical ESP32-CAM profile
        ```

        The held-out test split is not used for pruning, sensitivity, threshold selection, or
        candidate selection.
        """
    ),
    md(
        """
        ## Paper contract and deliberate adaptations

        Primary references:

        - Han et al., [*Learning both Weights and Connections for Efficient Neural Networks*](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf)
        - Zhu and Gupta, [*To Prune, or Not to Prune*](https://arxiv.org/abs/1710.01878)

        Han et al. establish the train → magnitude-prune → retrain procedure, preserve surviving
        weights, use layer sensitivity to choose different thresholds, lower the retraining
        learning rate, and repeat pruning/retraining. This notebook implements those concepts with
        registered Fast-80-specific choices:

        | Decision | Notebook 13 implementation |
        |---|---|
        | Granularity | individual Conv2D, DepthwiseConv2D, and Dense kernel scalars |
        | Criterion | absolute magnitude, `|w|` |
        | Layer sensitivity | validation-subset PR-AUC and BCE damage after 50% local pruning |
        | Threshold policy | sensitivity-adjusted magnitude scores, exact global target |
        | Iteration | five cumulative prune/retrain rounds |
        | Recovery rate | `1e-5`, one-tenth of Fast-80's `1e-4` |
        | Mask rule | removed connections never regrow |
        | Runtime | stock dense TFLite Micro / ESP-NN; no sparse acceleration claim |

        Gzip and packed sparse estimates are storage diagnostics. Only a physical-device capture
        can establish latency, tensor-arena, SRAM, PSRAM, or end-to-end FPS.
        """
    ),
    md("## 1 · Locate the repository deterministically"),
    code(
        """
        from pathlib import Path

        ROOT = Path.cwd().resolve()
        if not (ROOT / "pyproject.toml").exists():
            ROOT = ROOT.parent.parent.resolve()
        assert (ROOT / "pyproject.toml").exists(), f"Repository root not found from {Path.cwd()}"
        print(ROOT)
        """
    ),
    md("## 2 · Import deterministic experiment dependencies"),
    code(
        """
        import gzip
        import hashlib
        import json
        import math
        import os
        import platform
        import time

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import tensorflow as tf
        from IPython.display import Markdown, display
        from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, log_loss

        from vww_esp32.evaluation import (
            classification_metrics,
            expected_calibration_error,
            select_threshold,
        )
        from vww_esp32.exporting import convert_full_integer, inspect_tflite
        from vww_esp32.preprocessing import make_dataset, representative_dataset, split_manifest
        from vww_esp32.profiling import profile_model

        SEED = 42
        np.random.seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_columns", 80)
        print({
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "devices": [device.device_type for device in tf.config.list_physical_devices()],
        })
        """
    ),
    md("## 3 · Register the complete experiment before looking at results"),
    code(
        """
        SPARSITY_LEVELS = (0.25, 0.50, 0.75, 0.90)
        ONE_SHOT_EPOCHS = 2
        ONE_SHOT_LEARNING_RATE = 5e-5
        SENSITIVITY_LOCAL_SPARSITY = 0.50
        SENSITIVITY_BATCHES = 8
        SENSITIVITY_PROTECTION_STRENGTH = 20.0
        ITERATIVE_ROUNDS = 5
        ROUND_RECOVERY_EPOCHS = 1
        FINAL_RECOVERY_MAX_EPOCHS = 5
        FINAL_RECOVERY_PATIENCE = 2
        ITERATIVE_LEARNING_RATE = 1e-5
        BATCH_SIZE = 64
        IMAGE_SIZE = (80, 80)
        REPRESENTATIVE_SAMPLES = 500
        MINIMUM_RECALL = 0.80
        MAXIMUM_F1_DROP = 0.03
        HOST_WARMUPS = 20
        HOST_RUNS = 100
        REUSE_CACHE = os.environ.get("VWW_REUSE_PRUNING_CACHE", "0") == "1"

        FLOAT_MODEL_PATH = ROOT / "artifacts/fast_80/models/final.keras"
        DEPLOYED_BASELINE_TFLITE_PATH = ROOT / "artifacts/fast_80/models/vww_mobilenetv1_80_int8.tflite"
        MANIFEST_PATH = ROOT / "data/processed/manifest.csv"
        REFERENCE_CONTRACT_PATH = ROOT / "artifacts/pruning/reference_audit/baseline_contract.json"
        BASELINE_DEVICE_DIR = ROOT / "artifacts/device_profiles/fast_80"

        OUTPUT_DIR = ROOT / "artifacts/pruning/unstructured_magnitude"
        MODEL_DIR = OUTPUT_DIR / "models"
        REPORT_DIR = OUTPUT_DIR / "reports"
        FIGURE_DIR = OUTPUT_DIR / "figures"
        for directory in (OUTPUT_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        required = [FLOAT_MODEL_PATH, DEPLOYED_BASELINE_TFLITE_PATH, MANIFEST_PATH, REFERENCE_CONTRACT_PATH]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
        assert not missing, f"Missing required inputs: {missing}"
        print({"sparsities": SPARSITY_LEVELS, "reuse_cache": REUSE_CACHE})
        """
    ),
    md("## 4 · Verify the frozen Fast-80 pruning baseline"),
    code(
        """
        def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_bytes), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        with REFERENCE_CONTRACT_PATH.open(encoding="utf-8") as handle:
            reference_contract = json.load(handle)

        current_hashes = {
            str(FLOAT_MODEL_PATH.relative_to(ROOT)): sha256_file(FLOAT_MODEL_PATH),
            str(DEPLOYED_BASELINE_TFLITE_PATH.relative_to(ROOT)): sha256_file(DEPLOYED_BASELINE_TFLITE_PATH),
            str(MANIFEST_PATH.relative_to(ROOT)): sha256_file(MANIFEST_PATH),
        }
        for path, digest in current_hashes.items():
            assert reference_contract["hashes"][path] == digest, f"Frozen input changed: {path}"
        display(pd.DataFrame([{"artifact": path, "sha256": digest} for path, digest in current_hashes.items()]).style.hide(axis="index"))
        """
    ),
    md("## 5 · Load and profile Fast-80 before constructing masks"),
    code(
        """
        baseline_model = tf.keras.models.load_model(FLOAT_MODEL_PATH, compile=False)
        baseline_layers, baseline_liveness, baseline_summary = profile_model(baseline_model, batch_size=1)
        assert baseline_model.count_params() == reference_contract["float_parameters"]
        assert int(baseline_summary["estimated_macs_batch1"]) == reference_contract["dense_macs_batch1"]
        print({
            "input": baseline_model.input_shape,
            "parameters": baseline_model.count_params(),
            "dense_macs": int(baseline_summary["estimated_macs_batch1"]),
            "tflite_live_peak_bytes": reference_contract["tflite_live_activation_peak_bytes"],
        })
        """
    ),
    md("## 6 · Load the frozen train, validation, and test partitions"),
    code(
        """
        manifest = pd.read_csv(MANIFEST_PATH)
        splits = split_manifest(manifest)
        train_ds = make_dataset(splits["train"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=True, seed=SEED)
        val_ds = make_dataset(splits["val"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=False)
        test_ds = make_dataset(splits["test"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=False)
        print({name: len(frame) for name, frame in splits.items()})
        """
    ),
    md("## 7 · Define threshold and evaluation helpers"),
    code(
        """
        def collect_predictions_quiet(model: tf.keras.Model, dataset) -> tuple[np.ndarray, np.ndarray]:
            labels = np.concatenate([np.asarray(label).reshape(-1) for _, label in dataset]).astype(int)
            probabilities = np.asarray(model.predict(dataset, verbose=0)).reshape(-1)
            return labels, probabilities


        def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
            threshold_result = select_threshold(
                labels,
                probabilities,
                objective="f1",
                minimum_recall=MINIMUM_RECALL,
                false_positive_cost=1.0,
                false_negative_cost=2.0,
            )
            metrics = classification_metrics(labels, probabilities, threshold_result["threshold"])
            metrics["expected_calibration_error_10_bins"] = expected_calibration_error(labels, probabilities, bins=10)
            return metrics
        """
    ),
    md("## 8 · Recompute the validation reference in this exact runtime"),
    code(
        """
        val_labels, baseline_val_probabilities = collect_predictions_quiet(baseline_model, val_ds)
        baseline_val_metrics = evaluate_probabilities(val_labels, baseline_val_probabilities)
        print({key: baseline_val_metrics[key] for key in ("threshold", "accuracy", "precision", "recall", "f1", "pr_auc")})
        """
    ),
    md("## 9 · Identify the exact fine-grained pruning domain"),
    code(
        """
        PRUNABLE_TYPES = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)


        def prunable_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
            return [layer for layer in model.layers if isinstance(layer, PRUNABLE_TYPES)]


        prunable_inventory = pd.DataFrame([
            {
                "layer": layer.name,
                "type": layer.__class__.__name__,
                "kernel_shape": "×".join(map(str, layer.kernel.shape)),
                "kernel_weights": int(np.prod(layer.kernel.shape)),
            }
            for layer in prunable_layers(baseline_model)
        ])
        prunable_inventory.to_csv(REPORT_DIR / "prunable_tensor_inventory.csv", index=False)
        print(f"{len(prunable_inventory)} kernels; {prunable_inventory['kernel_weights'].sum():,} scalar weights")
        display(prunable_inventory.style.hide(axis="index"))
        """
    ),
    md("## 10 · Implement exact masks and permanent zero enforcement"),
    code(
        """
        def split_flat_mask(model: tf.keras.Model, flat_mask: np.ndarray) -> dict[str, np.ndarray]:
            masks, offset = {}, 0
            for layer in prunable_layers(model):
                size = int(np.prod(layer.kernel.shape))
                masks[layer.name] = flat_mask[offset:offset + size].reshape(layer.kernel.shape).astype(np.float32)
                offset += size
            assert offset == flat_mask.size
            return masks


        def flatten_masks(model: tf.keras.Model, masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
            if masks is None:
                return np.ones(sum(int(np.prod(layer.kernel.shape)) for layer in prunable_layers(model)), dtype=np.float32)
            return np.concatenate([masks[layer.name].reshape(-1) for layer in prunable_layers(model)]).astype(np.float32)


        def build_global_magnitude_masks(model: tf.keras.Model, target_sparsity: float) -> dict[str, np.ndarray]:
            magnitudes = np.concatenate([np.abs(np.asarray(layer.kernel.numpy())).reshape(-1) for layer in prunable_layers(model)])
            prune_count = int(np.floor(magnitudes.size * target_sparsity))
            order = np.argsort(magnitudes, kind="stable")
            flat_mask = np.ones(magnitudes.size, dtype=np.float32)
            flat_mask[order[:prune_count]] = 0.0
            return split_flat_mask(model, flat_mask)


        def apply_masks(model: tf.keras.Model, masks: dict[str, np.ndarray]) -> None:
            for layer_name, mask in masks.items():
                layer = model.get_layer(layer_name)
                layer.kernel.assign(layer.kernel * tf.convert_to_tensor(mask, dtype=layer.kernel.dtype))


        def measured_kernel_sparsity(model: tf.keras.Model) -> float:
            kernels = [np.asarray(layer.kernel.numpy()) for layer in prunable_layers(model)]
            return float(sum(np.count_nonzero(kernel == 0) for kernel in kernels) / sum(kernel.size for kernel in kernels))


        class EnforceMasks(tf.keras.callbacks.Callback):
            def __init__(self, masks: dict[str, np.ndarray]):
                super().__init__()
                self.masks = masks

            def on_train_batch_end(self, batch, logs=None):
                apply_masks(self.model, self.masks)

            def on_epoch_end(self, epoch, logs=None):
                apply_masks(self.model, self.masks)
        """
    ),
    md("## 11 · Compile a pruned model without letting BatchNorm drift"),
    code(
        """
        def compile_for_recovery(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                ],
            )
            return model
        """
    ),
    md("## 12 · Define the global one-shot control"),
    code(
        """
        def run_one_shot_candidate(target_sparsity: float) -> tuple[dict[str, object], pd.DataFrame]:
            tag = f"s{int(round(target_sparsity * 100)):02d}"
            model = tf.keras.models.load_model(FLOAT_MODEL_PATH, compile=False)
            masks = build_global_magnitude_masks(model, target_sparsity)
            apply_masks(model, masks)
            _, immediate_probabilities = collect_predictions_quiet(model, val_ds)
            immediate = evaluate_probabilities(val_labels, immediate_probabilities)

            tf.keras.utils.set_random_seed(SEED)
            candidate_train_ds = make_dataset(splits["train"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=True, seed=SEED)
            model = compile_for_recovery(model, ONE_SHOT_LEARNING_RATE)
            started = time.perf_counter()
            history = model.fit(
                candidate_train_ds,
                validation_data=val_ds,
                epochs=ONE_SHOT_EPOCHS,
                callbacks=[EnforceMasks(masks)],
                verbose=2,
            )
            elapsed = time.perf_counter() - started
            apply_masks(model, masks)
            _, recovered_probabilities = collect_predictions_quiet(model, val_ds)
            recovered = evaluate_probabilities(val_labels, recovered_probabilities)
            actual = measured_kernel_sparsity(model)
            assert abs(actual - target_sparsity) < 1e-5

            model_path = MODEL_DIR / f"one_shot_{tag}_recovered.keras"
            mask_path = MODEL_DIR / f"one_shot_{tag}_masks.npz"
            model.save(model_path)
            np.savez_compressed(mask_path, **masks)
            detail = {
                "strategy": "one_shot_global",
                "tag": tag,
                "target_sparsity": target_sparsity,
                "actual_sparsity": actual,
                "immediate_validation": immediate,
                "recovered_validation": recovered,
                "recovery_epochs": ONE_SHOT_EPOCHS,
                "recovery_seconds": elapsed,
                "model_path": str(model_path.relative_to(ROOT)),
                "mask_path": str(mask_path.relative_to(ROOT)),
            }
            (REPORT_DIR / f"one_shot_{tag}_metrics.json").write_text(json.dumps(detail, indent=2) + "\\n", encoding="utf-8")
            history_frame = pd.DataFrame(history.history)
            history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
            history_frame.insert(0, "tag", tag)
            history_frame.insert(0, "strategy", "one_shot_global")
            row = {key: value for key, value in detail.items() if key not in ("immediate_validation", "recovered_validation")}
            for prefix, metrics in (("immediate", immediate), ("recovered", recovered)):
                for key, value in metrics.items():
                    if key != "confusion_matrix":
                        row[f"{prefix}_{key}"] = value
            del model
            tf.keras.backend.clear_session()
            return row, history_frame
        """
    ),
    md("## 13 · Execute or reuse the one-shot sweep"),
    code(
        """
        one_shot_csv = REPORT_DIR / "one_shot_sweep.csv"
        one_shot_history_csv = REPORT_DIR / "one_shot_recovery_history.csv"
        one_shot_cache_ready = one_shot_csv.exists() and one_shot_history_csv.exists() and all(
            (MODEL_DIR / f"one_shot_s{int(level * 100):02d}_recovered.keras").exists() for level in SPARSITY_LEVELS
        )
        if REUSE_CACHE and one_shot_cache_ready:
            one_shot_sweep = pd.read_csv(one_shot_csv)
            one_shot_history = pd.read_csv(one_shot_history_csv)
            print("Reused one-shot cache")
        else:
            rows, histories = [], []
            for target in SPARSITY_LEVELS:
                print(f"\\n=== One-shot target {target:.0%} ===")
                row, history = run_one_shot_candidate(target)
                rows.append(row)
                histories.append(history)
            one_shot_sweep = pd.DataFrame(rows).sort_values("target_sparsity").reset_index(drop=True)
            one_shot_history = pd.concat(histories, ignore_index=True)
            one_shot_sweep.to_csv(one_shot_csv, index=False)
            one_shot_history.to_csv(one_shot_history_csv, index=False)
        display(one_shot_sweep[["tag", "actual_sparsity", "immediate_f1", "recovered_f1", "recovered_recall", "recovered_pr_auc"]].style.hide(axis="index").format({
            "actual_sparsity": "{:.1%}", "immediate_f1": "{:.3f}", "recovered_f1": "{:.3f}", "recovered_recall": "{:.3f}", "recovered_pr_auc": "{:.3f}"
        }))
        """
    ),
    md("## 14 · Measure layer sensitivity before assigning paper-inspired thresholds"),
    code(
        """
        sensitivity_csv = REPORT_DIR / "layer_sensitivity.csv"
        if REUSE_CACHE and sensitivity_csv.exists():
            layer_sensitivity = pd.read_csv(sensitivity_csv)
            print("Reused layer sensitivity cache")
        else:
            sensitivity_ds = val_ds.take(SENSITIVITY_BATCHES)
            sensitivity_labels, sensitivity_baseline_probabilities = collect_predictions_quiet(baseline_model, sensitivity_ds)
            baseline_probe_pr_auc = average_precision_score(sensitivity_labels, sensitivity_baseline_probabilities)
            baseline_probe_loss = log_loss(sensitivity_labels, np.clip(sensitivity_baseline_probabilities, 1e-7, 1 - 1e-7))
            rows = []
            for layer in prunable_layers(baseline_model):
                original = np.asarray(layer.kernel.numpy()).copy()
                flat = np.abs(original).reshape(-1)
                prune_count = int(np.floor(flat.size * SENSITIVITY_LOCAL_SPARSITY))
                local_mask = np.ones(flat.size, dtype=np.float32)
                local_mask[np.argsort(flat, kind="stable")[:prune_count]] = 0.0
                layer.kernel.assign(original * local_mask.reshape(original.shape))
                _, probabilities = collect_predictions_quiet(baseline_model, sensitivity_ds)
                pr_auc = average_precision_score(sensitivity_labels, probabilities)
                loss = log_loss(sensitivity_labels, np.clip(probabilities, 1e-7, 1 - 1e-7))
                layer.kernel.assign(original)
                rows.append({
                    "layer": layer.name,
                    "type": layer.__class__.__name__,
                    "kernel_weights": flat.size,
                    "probe_sparsity": SENSITIVITY_LOCAL_SPARSITY,
                    "baseline_pr_auc": baseline_probe_pr_auc,
                    "pruned_pr_auc": pr_auc,
                    "pr_auc_drop": max(0.0, baseline_probe_pr_auc - pr_auc),
                    "baseline_bce": baseline_probe_loss,
                    "pruned_bce": loss,
                    "relative_bce_increase": max(0.0, (loss - baseline_probe_loss) / max(baseline_probe_loss, 1e-8)),
                })
            layer_sensitivity = pd.DataFrame(rows)
            layer_sensitivity["risk_raw"] = layer_sensitivity["pr_auc_drop"] + 0.25 * layer_sensitivity["relative_bce_increase"]
            maximum_risk = max(float(layer_sensitivity["risk_raw"].max()), 1e-12)
            layer_sensitivity["risk_normalized"] = layer_sensitivity["risk_raw"] / maximum_risk
            layer_sensitivity["protection_multiplier"] = 1.0 + SENSITIVITY_PROTECTION_STRENGTH * layer_sensitivity["risk_normalized"]
            layer_sensitivity.to_csv(sensitivity_csv, index=False)
        layer_protection = dict(zip(layer_sensitivity["layer"], layer_sensitivity["protection_multiplier"]))
        display(layer_sensitivity.sort_values("risk_raw", ascending=False).style.hide(axis="index").format({
            "probe_sparsity": "{:.0%}", "baseline_pr_auc": "{:.3f}", "pruned_pr_auc": "{:.3f}", "pr_auc_drop": "{:.3f}",
            "baseline_bce": "{:.3f}", "pruned_bce": "{:.3f}", "relative_bce_increase": "{:.1%}", "risk_raw": "{:.3f}",
            "risk_normalized": "{:.2f}", "protection_multiplier": "{:.1f}×"
        }))
        """
    ),
    md("## 15 · Visualize which layers require smaller effective thresholds"),
    code(
        """
        sensitivity_plot = layer_sensitivity.sort_values("protection_multiplier").tail(14)
        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(sensitivity_plot["layer"], sensitivity_plot["protection_multiplier"], color="#2878B5")
        ax.set(title="Layer sensitivity controls the effective pruning threshold", xlabel="Magnitude protection multiplier (×)", ylabel="")
        for bar, value in zip(bars, sensitivity_plot["protection_multiplier"]):
            ax.text(value + 0.15, bar.get_y() + bar.get_height() / 2, f"{value:.1f}×", va="center", fontsize=9)
        figure_path = FIGURE_DIR / "layer_sensitivity_protection.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 16 · Extend masks using sensitivity-adjusted magnitude scores"),
    code(
        """
        def extend_sensitivity_adjusted_masks(
            model: tf.keras.Model,
            target_sparsity: float,
            current_masks: dict[str, np.ndarray] | None,
            protection: dict[str, float],
        ) -> dict[str, np.ndarray]:
            current_flat = flatten_masks(model, current_masks)
            adjusted_scores = []
            for layer in prunable_layers(model):
                magnitude = np.abs(np.asarray(layer.kernel.numpy())).reshape(-1)
                adjusted_scores.append(magnitude * float(protection[layer.name]))
            scores = np.concatenate(adjusted_scores)
            total = scores.size
            desired_zeros = int(np.floor(total * target_sparsity))
            current_zeros = int(np.count_nonzero(current_flat == 0))
            new_prune_count = max(0, desired_zeros - current_zeros)
            scores[current_flat == 0] = np.inf
            active_order = np.argsort(scores, kind="stable")
            current_flat[active_order[:new_prune_count]] = 0.0
            masks = split_flat_mask(model, current_flat)
            assert np.count_nonzero(current_flat == 0) == desired_zeros
            return masks
        """
    ),
    md("## 17 · Define five-round paper-inspired prune/retrain"),
    code(
        """
        def run_iterative_candidate(target_sparsity: float) -> tuple[dict[str, object], pd.DataFrame]:
            tag = f"s{int(round(target_sparsity * 100)):02d}"
            model = tf.keras.models.load_model(FLOAT_MODEL_PATH, compile=False)
            model = compile_for_recovery(model, ITERATIVE_LEARNING_RATE)
            masks = None
            round_rows = []
            tf.keras.utils.set_random_seed(SEED)
            candidate_train_ds = make_dataset(splits["train"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=True, seed=SEED)
            started = time.perf_counter()

            for round_index, cumulative_target in enumerate(np.linspace(target_sparsity / ITERATIVE_ROUNDS, target_sparsity, ITERATIVE_ROUNDS), start=1):
                masks = extend_sensitivity_adjusted_masks(model, float(cumulative_target), masks, layer_protection)
                apply_masks(model, masks)
                _, immediate_probabilities = collect_predictions_quiet(model, val_ds)
                immediate = evaluate_probabilities(val_labels, immediate_probabilities)
                history = model.fit(
                    candidate_train_ds,
                    validation_data=val_ds,
                    epochs=ROUND_RECOVERY_EPOCHS,
                    callbacks=[EnforceMasks(masks)],
                    verbose=0,
                )
                apply_masks(model, masks)
                _, recovered_probabilities = collect_predictions_quiet(model, val_ds)
                recovered = evaluate_probabilities(val_labels, recovered_probabilities)
                round_rows.append({
                    "strategy": "paper_iterative",
                    "tag": tag,
                    "round": round_index,
                    "cumulative_target": float(cumulative_target),
                    "actual_sparsity": measured_kernel_sparsity(model),
                    "immediate_f1": immediate["f1"],
                    "immediate_pr_auc": immediate["pr_auc"],
                    "recovered_f1": recovered["f1"],
                    "recovered_pr_auc": recovered["pr_auc"],
                    "train_loss": history.history["loss"][-1],
                    "val_pr_auc_fit": history.history["val_pr_auc"][-1],
                })

            _, final_immediate_probabilities = collect_predictions_quiet(model, val_ds)
            final_immediate = evaluate_probabilities(val_labels, final_immediate_probabilities)
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=FINAL_RECOVERY_PATIENCE,
                restore_best_weights=True,
            )
            final_history = model.fit(
                candidate_train_ds,
                validation_data=val_ds,
                epochs=FINAL_RECOVERY_MAX_EPOCHS,
                callbacks=[EnforceMasks(masks), early_stop],
                verbose=2,
            )
            apply_masks(model, masks)
            elapsed = time.perf_counter() - started
            _, recovered_probabilities = collect_predictions_quiet(model, val_ds)
            recovered = evaluate_probabilities(val_labels, recovered_probabilities)
            actual = measured_kernel_sparsity(model)
            assert abs(actual - target_sparsity) < 1e-5

            model_path = MODEL_DIR / f"iterative_{tag}_recovered.keras"
            mask_path = MODEL_DIR / f"iterative_{tag}_masks.npz"
            model.save(model_path)
            np.savez_compressed(mask_path, **masks)
            detail = {
                "strategy": "paper_iterative",
                "tag": tag,
                "target_sparsity": target_sparsity,
                "actual_sparsity": actual,
                "immediate_validation": final_immediate,
                "recovered_validation": recovered,
                "prune_retrain_rounds": ITERATIVE_ROUNDS,
                "final_recovery_epochs": len(final_history.history["loss"]),
                "recovery_seconds": elapsed,
                "model_path": str(model_path.relative_to(ROOT)),
                "mask_path": str(mask_path.relative_to(ROOT)),
            }
            (REPORT_DIR / f"iterative_{tag}_metrics.json").write_text(json.dumps(detail, indent=2) + "\\n", encoding="utf-8")
            row = {key: value for key, value in detail.items() if key not in ("immediate_validation", "recovered_validation")}
            for prefix, metrics in (("immediate", final_immediate), ("recovered", recovered)):
                for key, value in metrics.items():
                    if key != "confusion_matrix":
                        row[f"{prefix}_{key}"] = value
            del model
            tf.keras.backend.clear_session()
            return row, pd.DataFrame(round_rows)
        """
    ),
    md("## 18 · Execute or reuse the iterative sweep"),
    code(
        """
        iterative_csv = REPORT_DIR / "iterative_sweep.csv"
        iterative_round_csv = REPORT_DIR / "iterative_round_history.csv"
        iterative_cache_ready = iterative_csv.exists() and iterative_round_csv.exists() and all(
            (MODEL_DIR / f"iterative_s{int(level * 100):02d}_recovered.keras").exists() for level in SPARSITY_LEVELS
        )
        if REUSE_CACHE and iterative_cache_ready:
            iterative_sweep = pd.read_csv(iterative_csv)
            iterative_round_history = pd.read_csv(iterative_round_csv)
            print("Reused iterative cache")
        else:
            rows, round_frames = [], []
            for target in SPARSITY_LEVELS:
                print(f"\\n=== Paper-inspired iterative target {target:.0%} ===")
                row, rounds = run_iterative_candidate(target)
                rows.append(row)
                round_frames.append(rounds)
            iterative_sweep = pd.DataFrame(rows).sort_values("target_sparsity").reset_index(drop=True)
            iterative_round_history = pd.concat(round_frames, ignore_index=True)
            iterative_sweep.to_csv(iterative_csv, index=False)
            iterative_round_history.to_csv(iterative_round_csv, index=False)
        display(iterative_sweep[["tag", "actual_sparsity", "immediate_f1", "recovered_f1", "recovered_recall", "recovered_pr_auc", "final_recovery_epochs"]].style.hide(axis="index").format({
            "actual_sparsity": "{:.1%}", "immediate_f1": "{:.3f}", "recovered_f1": "{:.3f}", "recovered_recall": "{:.3f}", "recovered_pr_auc": "{:.3f}"
        }))
        """
    ),
    md("## 19 · Compare one-shot and iterative validation quality"),
    code(
        """
        float_sweep = pd.concat([one_shot_sweep, iterative_sweep], ignore_index=True)
        float_sweep["f1_drop_vs_baseline"] = baseline_val_metrics["f1"] - float_sweep["recovered_f1"]
        float_sweep["recall_gate"] = float_sweep["recovered_recall"] >= MINIMUM_RECALL
        float_sweep["f1_gate"] = float_sweep["f1_drop_vs_baseline"] <= MAXIMUM_F1_DROP
        float_sweep["eligible_float"] = float_sweep["recall_gate"] & float_sweep["f1_gate"]
        float_sweep.to_csv(REPORT_DIR / "float_strategy_comparison.csv", index=False)

        labels = {"one_shot_global": "Global one-shot", "paper_iterative": "Paper-inspired iterative"}
        colors = {"one_shot_global": "#E15759", "paper_iterative": "#2878B5"}
        markers = {"one_shot_global": "o", "paper_iterative": "s"}
        metrics = [("recovered_f1", "F1"), ("recovered_recall", "Recall"), ("recovered_pr_auc", "PR-AUC")]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for ax, (metric, title) in zip(axes, metrics):
            for strategy, group in float_sweep.groupby("strategy"):
                ax.plot(group["actual_sparsity"] * 100, group[metric], marker=markers[strategy], color=colors[strategy], label=labels[strategy])
            ax.axhline(baseline_val_metrics[metric.replace("recovered_", "")], color="#444444", linewidth=1.3, linestyle="--", label="Fast-80 pruning baseline")
            values = np.r_[float_sweep[metric], baseline_val_metrics[metric.replace("recovered_", "")]]
            ax.set(title=title, xlabel="Kernel sparsity (%)", ylabel="Validation score", ylim=(max(0, values.min() - 0.06), min(1.02, values.max() + 0.06)))
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
        fig.suptitle("Magnitude pruning: global one-shot versus paper-inspired iterative recovery", fontsize=16, weight="bold", y=1.17)
        figure_path = FIGURE_DIR / "one_shot_vs_iterative_quality.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 20 · Visualize recovery after every iterative pruning event"),
    code(
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, sharex=True)
        for ax, (tag, group) in zip(axes.flat, iterative_round_history.groupby("tag", sort=True)):
            ax.plot(group["cumulative_target"] * 100, group["immediate_f1"], "o--", color="#E15759", label="Immediately after prune")
            ax.plot(group["cumulative_target"] * 100, group["recovered_f1"], "s-", color="#2878B5", label="After round recovery")
            ax.axhline(baseline_val_metrics["f1"], color="#444444", linewidth=1.2, linestyle=":", label="Fast-80")
            ax.set(title=f"Final target {tag[1:]}%", xlabel="Cumulative sparsity (%)", ylabel="Validation F1")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
        fig.suptitle("Damage and healing across five prune/retrain rounds", fontsize=16, weight="bold", y=1.08)
        figure_path = FIGURE_DIR / "iterative_round_recovery.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 21 · Compare where each policy places zeros"),
    code(
        """
        layer_sparsity_rows = []
        for strategy, prefix in (("one_shot_global", "one_shot"), ("paper_iterative", "iterative")):
            for target in SPARSITY_LEVELS:
                tag = f"s{int(target * 100):02d}"
                model = tf.keras.models.load_model(MODEL_DIR / f"{prefix}_{tag}_recovered.keras", compile=False)
                for layer in prunable_layers(model):
                    kernel = np.asarray(layer.kernel.numpy())
                    layer_sparsity_rows.append({"strategy": strategy, "tag": tag, "layer": layer.name, "layer_sparsity": float(np.mean(kernel == 0)), "kernel_weights": kernel.size})
                del model
        layer_sparsity = pd.DataFrame(layer_sparsity_rows)
        layer_sparsity.to_csv(REPORT_DIR / "layer_sparsity_by_strategy.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(15, 11), constrained_layout=True, sharey=True)
        for ax, strategy in zip(axes, ("one_shot_global", "paper_iterative")):
            matrix = layer_sparsity[layer_sparsity["strategy"] == strategy].pivot(index="layer", columns="tag", values="layer_sparsity")
            sns.heatmap(matrix, cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".0%", linewidths=0.3, cbar=strategy == "paper_iterative", cbar_kws={"label": "Exact layer sparsity"}, ax=ax)
            ax.set(title=labels[strategy], xlabel="Global target", ylabel="" if strategy == "paper_iterative" else "Layer")
        fig.suptitle("Sensitivity weighting protects fragile layers without changing total sparsity", fontsize=16, weight="bold", y=1.02)
        figure_path = FIGURE_DIR / "layer_sparsity_policy_comparison.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 22 · Export the baseline and every candidate through one identical full-INT8 path"),
    code(
        """
        export_specs = [{"strategy": "baseline", "tag": "s00", "target_sparsity": 0.0, "keras_path": FLOAT_MODEL_PATH}]
        for row in float_sweep.itertuples(index=False):
            export_specs.append({"strategy": row.strategy, "tag": row.tag, "target_sparsity": row.target_sparsity, "keras_path": ROOT / row.model_path})

        export_rows, model_paths = [], {}
        for spec in export_specs:
            export_key = f"{spec['strategy']}__{spec['tag']}"
            if spec["strategy"] == "baseline":
                tflite_path = MODEL_DIR / "fast80_control_reexport_int8.tflite"
            else:
                prefix = "one_shot" if spec["strategy"] == "one_shot_global" else "iterative"
                tflite_path = MODEL_DIR / f"{prefix}_{spec['tag']}_int8.tflite"
            if not (REUSE_CACHE and tflite_path.exists()):
                candidate_model = tf.keras.models.load_model(spec["keras_path"], compile=False)
                representative = representative_dataset(splits["train"], image_size=IMAGE_SIZE, samples=REPRESENTATIVE_SAMPLES, seed=SEED)
                convert_full_integer(candidate_model, representative, tflite_path)
                del candidate_model
                tf.keras.backend.clear_session()
            info = inspect_tflite(tflite_path)
            assert info["input"]["dtype"] == "int8" and info["output"]["dtype"] == "int8"
            model_paths[export_key] = tflite_path
            export_rows.append({
                "export_key": export_key,
                "strategy": spec["strategy"],
                "tag": spec["tag"],
                "target_sparsity": spec["target_sparsity"],
                "tflite_path": str(tflite_path.relative_to(ROOT)),
                "tflite_bytes": tflite_path.stat().st_size,
                "gzip_bytes": len(gzip.compress(tflite_path.read_bytes(), compresslevel=9)),
                "operators": len(info["operators"]),
                "input_dtype": info["input"]["dtype"],
                "output_dtype": info["output"]["dtype"],
                "sha256": sha256_file(tflite_path),
            })
        exports = pd.DataFrame(export_rows)
        exports.to_csv(REPORT_DIR / "int8_export_inventory.csv", index=False)
        display(exports.style.hide(axis="index").format({"target_sparsity": "{:.0%}", "tflite_bytes": "{:,}", "gzip_bytes": "{:,}"}))
        """
    ),
    md("## 23 · Evaluate every INT8 candidate on validation before selection"),
    code(
        """
        def collect_tflite_predictions(model_path: Path, dataset) -> tuple[np.ndarray, np.ndarray]:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            input_scale, input_zero = input_detail["quantization"]
            output_scale, output_zero = output_detail["quantization"]
            labels_out, probabilities = [], []
            for image_batch, label_batch in dataset:
                labels_out.extend(np.asarray(label_batch).reshape(-1).astype(int).tolist())
                for image in np.asarray(image_batch):
                    quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)[None, ...]
                    interpreter.set_tensor(input_detail["index"], quantized)
                    interpreter.invoke()
                    raw = interpreter.get_tensor(output_detail["index"])[0, 0]
                    probabilities.append((float(raw) - output_zero) * output_scale)
            return np.asarray(labels_out), np.asarray(probabilities)


        int8_rows, int8_validation_details = [], {}
        for row in exports.itertuples(index=False):
            labels_out, probabilities = collect_tflite_predictions(ROOT / row.tflite_path, val_ds)
            metrics = evaluate_probabilities(labels_out, probabilities)
            int8_validation_details[row.export_key] = metrics
            int8_rows.append({
                "export_key": row.export_key,
                "strategy": row.strategy,
                "tag": row.tag,
                "target_sparsity": row.target_sparsity,
                **{key: metrics[key] for key in ("threshold", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "predicted_positive_rate", "expected_calibration_error_10_bins")},
            })
        int8_metrics = pd.DataFrame(int8_rows)
        int8_metrics.to_csv(REPORT_DIR / "int8_validation_metrics.csv", index=False)
        (REPORT_DIR / "int8_validation_metrics.json").write_text(json.dumps(int8_validation_details, indent=2) + "\\n", encoding="utf-8")
        display(int8_metrics.style.hide(axis="index").format({key: "{:.3f}" for key in ("threshold", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "predicted_positive_rate", "expected_calibration_error_10_bins")}))
        """
    ),
    md("## 24 · Select the deployable paper-inspired candidate using INT8 validation gates"),
    code(
        """
        baseline_int8_val = int8_metrics[int8_metrics["strategy"] == "baseline"].iloc[0]
        iterative_int8 = int8_metrics[int8_metrics["strategy"] == "paper_iterative"].copy()
        iterative_int8["f1_drop_vs_baseline"] = baseline_int8_val["f1"] - iterative_int8["f1"]
        iterative_int8["recall_gate"] = iterative_int8["recall"] >= MINIMUM_RECALL
        iterative_int8["f1_gate"] = iterative_int8["f1_drop_vs_baseline"] <= MAXIMUM_F1_DROP
        iterative_int8["eligible"] = iterative_int8["recall_gate"] & iterative_int8["f1_gate"]
        if iterative_int8["eligible"].any():
            selected_int8_row = iterative_int8[iterative_int8["eligible"]].sort_values(["target_sparsity", "f1"], ascending=[False, False]).iloc[0]
            selection_reason = "highest paper-inspired sparsity satisfying INT8 validation recall and F1 gates"
        else:
            selected_int8_row = iterative_int8.sort_values(["f1", "target_sparsity"], ascending=[False, False]).iloc[0]
            selection_reason = "fallback: strongest paper-inspired INT8 validation F1"

        SELECTED_TAG = str(selected_int8_row["tag"])
        SELECTED_SPARSITY = float(selected_int8_row["target_sparsity"])
        SELECTED_EXPORT_KEY = str(selected_int8_row["export_key"])
        SELECTED_TFLITE_PATH = ROOT / exports.set_index("export_key").loc[SELECTED_EXPORT_KEY, "tflite_path"]
        SELECTED_FLOAT_PATH = MODEL_DIR / f"iterative_{SELECTED_TAG}_recovered.keras"
        selected_float_row = iterative_sweep.set_index("tag").loc[SELECTED_TAG]
        print({"selected": SELECTED_EXPORT_KEY, "sparsity": SELECTED_SPARSITY, "reason": selection_reason, "tflite": str(SELECTED_TFLITE_PATH.relative_to(ROOT))})
        display(iterative_int8[["tag", "target_sparsity", "f1", "recall", "pr_auc", "f1_drop_vs_baseline", "eligible"]].style.hide(axis="index").format({
            "target_sparsity": "{:.0%}", "f1": "{:.3f}", "recall": "{:.3f}", "pr_auc": "{:.3f}", "f1_drop_vs_baseline": "{:+.3f}"
        }))
        """
    ),
    md("## 25 · Open the held-out test exactly once after selection"),
    code(
        """
        baseline_control_tflite = model_paths["baseline__s00"]
        test_labels, baseline_test_probabilities = collect_tflite_predictions(baseline_control_tflite, test_ds)
        _, selected_test_probabilities = collect_tflite_predictions(SELECTED_TFLITE_PATH, test_ds)
        baseline_test_metrics = classification_metrics(test_labels, baseline_test_probabilities, float(baseline_int8_val["threshold"]))
        baseline_test_metrics["expected_calibration_error_10_bins"] = expected_calibration_error(test_labels, baseline_test_probabilities, bins=10)
        selected_test_metrics = classification_metrics(test_labels, selected_test_probabilities, float(selected_int8_row["threshold"]))
        selected_test_metrics["expected_calibration_error_10_bins"] = expected_calibration_error(test_labels, selected_test_probabilities, bins=10)
        selected_float_model = tf.keras.models.load_model(SELECTED_FLOAT_PATH, compile=False)
        _, selected_float_test_probabilities = collect_predictions_quiet(selected_float_model, test_ds)
        selected_test_metrics["mean_absolute_probability_delta_vs_float"] = float(np.mean(np.abs(selected_test_probabilities - selected_float_test_probabilities)))
        test_details = {"baseline_int8": baseline_test_metrics, "selected_int8": selected_test_metrics}
        (REPORT_DIR / "selected_int8_test_metrics.json").write_text(json.dumps(test_details, indent=2) + "\\n", encoding="utf-8")
        display(pd.DataFrame([
            {"model": "Fast-80 pruning baseline INT8", **{key: baseline_test_metrics[key] for key in ("threshold", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")}},
            {"model": f"Paper-inspired {SELECTED_TAG} INT8", **{key: selected_test_metrics[key] for key in ("threshold", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")}},
        ]).style.hide(axis="index").format({key: "{:.3f}" for key in ("threshold", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")}))
        """
    ),
    md("## 26 · Show the held-out decision errors directly"),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        for ax, title, metrics in [
            (axes[0], "Fast-80 pruning baseline INT8", baseline_test_metrics),
            (axes[1], f"Paper-inspired {SELECTED_TAG} INT8", selected_test_metrics),
        ]:
            ConfusionMatrixDisplay(confusion_matrix=np.asarray(metrics["confusion_matrix"]), display_labels=["No person", "Person"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_title(f"{title}\\nF1={metrics['f1']:.3f}, recall={metrics['recall']:.3f}")
        figure_path = FIGURE_DIR / "test_confusion_matrices_revised.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 27 · Account for dense work and theoretical sparse work separately"),
    code(
        """
        baseline_layer_lookup = baseline_layers.set_index("layer_name")


        def theoretical_nonzero_macs(model: tf.keras.Model) -> int:
            total = 0
            for layer in prunable_layers(model):
                nonzeros = int(np.count_nonzero(np.asarray(layer.kernel.numpy())))
                if isinstance(layer, tf.keras.layers.Dense):
                    reuse = 1
                else:
                    profile = baseline_layer_lookup.loc[layer.name]
                    reuse = int(profile["output_height"] * profile["output_width"])
                total += nonzeros * reuse
            return total


        resource_rows = []
        for export in exports.itertuples(index=False):
            if export.strategy == "baseline":
                model = baseline_model
                sparsity = 0.0
            else:
                prefix = "one_shot" if export.strategy == "one_shot_global" else "iterative"
                model = tf.keras.models.load_model(MODEL_DIR / f"{prefix}_{export.tag}_recovered.keras", compile=False)
                sparsity = measured_kernel_sparsity(model)
            total_weights = int(prunable_inventory["kernel_weights"].sum())
            nonzeros = int(round(total_weights * (1 - sparsity)))
            resource_rows.append({
                "export_key": export.export_key,
                "strategy": export.strategy,
                "tag": export.tag,
                "target_sparsity": export.target_sparsity,
                "kernel_sparsity": sparsity,
                "kernel_nonzeros": nonzeros,
                "physical_parameters": reference_contract["float_parameters"],
                "dense_executed_macs": reference_contract["dense_macs_batch1"],
                "theoretical_nonzero_macs": theoretical_nonzero_macs(model),
                "tflite_live_activation_peak_bytes": reference_contract["tflite_live_activation_peak_bytes"],
                "tflite_bytes": export.tflite_bytes,
                "gzip_bytes": export.gzip_bytes,
                "bitmap_packed_kernel_bytes": math.ceil(total_weights / 8) + nonzeros,
                "index16_packed_kernel_bytes": nonzeros * 3 + (len(prunable_inventory) + 1) * 4,
                "operators": export.operators,
            })
            if export.strategy != "baseline":
                del model
        resources = pd.DataFrame(resource_rows)
        resources["theoretical_mac_reduction"] = 1 - resources["theoretical_nonzero_macs"] / resources["dense_executed_macs"]
        resources.to_csv(REPORT_DIR / "resource_comparison_revised.csv", index=False)
        display(resources.style.hide(axis="index").format({
            "target_sparsity": "{:.0%}", "kernel_sparsity": "{:.1%}", "kernel_nonzeros": "{:,}", "physical_parameters": "{:,}",
            "dense_executed_macs": "{:,}", "theoretical_nonzero_macs": "{:,}", "tflite_live_activation_peak_bytes": "{:,}",
            "tflite_bytes": "{:,}", "gzip_bytes": "{:,}", "bitmap_packed_kernel_bytes": "{:,}", "index16_packed_kernel_bytes": "{:,}",
            "theoretical_mac_reduction": "{:.1%}"
        }))
        """
    ),
    md("## 28 · Run a controlled host timing check without calling it an ESP32 result"),
    code(
        """
        first_image = np.asarray(next(iter(val_ds))[0][0])


        def benchmark_host_tflite(model_path: Path) -> dict[str, float]:
            interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            scale, zero = input_detail["quantization"]
            quantized = np.clip(np.round(first_image / scale + zero), -128, 127).astype(np.int8)[None, ...]
            interpreter.set_tensor(input_detail["index"], quantized)
            for _ in range(HOST_WARMUPS):
                interpreter.invoke()
            samples = []
            for _ in range(HOST_RUNS):
                started = time.perf_counter_ns()
                interpreter.invoke()
                samples.append((time.perf_counter_ns() - started) / 1e6)
            return {"host_median_ms": float(np.median(samples)), "host_p95_ms": float(np.percentile(samples, 95))}


        host_timings = pd.DataFrame([
            {"export_key": row.export_key, "strategy": row.strategy, "tag": row.tag, **benchmark_host_tflite(ROOT / row.tflite_path)}
            for row in exports.itertuples(index=False)
        ])
        host_timings.to_csv(REPORT_DIR / "host_tflite_timings_revised.csv", index=False)
        display(host_timings.style.hide(axis="index").format({"host_median_ms": "{:.3f}", "host_p95_ms": "{:.3f}"}))
        """
    ),
    md("## 29 · Visualize storage formats instead of treating gzip as deployed flash"),
    code(
        """
        storage_plot = resources[resources["strategy"].isin(["baseline", "paper_iterative"])].sort_values("target_sparsity")
        fig, ax = plt.subplots(figsize=(11, 7))
        x = np.arange(len(storage_plot))
        width = 0.21
        ax.bar(x - 1.5 * width, storage_plot["tflite_bytes"] / 1024, width, label="Dense TFLite", color="#2878B5")
        ax.bar(x - 0.5 * width, storage_plot["gzip_bytes"] / 1024, width, label="Gzip diagnostic", color="#59A14F")
        ax.bar(x + 0.5 * width, storage_plot["bitmap_packed_kernel_bytes"] / 1024, width, label="Bitmap + INT8 values", color="#F28E2B")
        ax.bar(x + 1.5 * width, storage_plot["index16_packed_kernel_bytes"] / 1024, width, label="INT16 index + INT8 values", color="#B07AA1")
        ax.set(title="Dense deployment bytes versus hypothetical sparse encodings", xlabel="Paper-inspired target", ylabel="Storage (KiB)", xticks=x, xticklabels=["Fast-80"] + [f"{int(v * 100)}%" for v in storage_plot["target_sparsity"].iloc[1:]])
        ax.legend(frameon=False, ncol=2)
        figure_path = FIGURE_DIR / "sparse_storage_encodings.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 30 · Visualize quality, storage, compute, and activation honestly"),
    code(
        """
        comparison = resources.merge(int8_metrics[["export_key", "f1", "recall", "pr_auc"]], on="export_key").merge(host_timings[["export_key", "host_median_ms"]], on="export_key")
        comparison = comparison[comparison["strategy"].isin(["baseline", "paper_iterative"])].sort_values("target_sparsity")
        x = comparison["target_sparsity"] * 100
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        axes[0, 0].plot(x, comparison["f1"], "o-", label="INT8 F1", color="#2878B5")
        axes[0, 0].plot(x, comparison["recall"], "s-", label="INT8 recall", color="#F28E2B")
        axes[0, 0].set(title="A · Validation quality", xlabel="Kernel sparsity (%)", ylabel="Score", ylim=(0, 1.02))
        axes[0, 0].legend(frameon=False)
        axes[0, 1].plot(x, comparison["tflite_bytes"] / 1024, "o-", label="Raw TFLite", color="#2878B5")
        axes[0, 1].plot(x, comparison["gzip_bytes"] / 1024, "s-", label="Gzip diagnostic", color="#59A14F")
        axes[0, 1].set(title="B · Serialized storage", xlabel="Kernel sparsity (%)", ylabel="KiB")
        axes[0, 1].legend(frameon=False)
        axes[1, 0].plot(x, comparison["dense_executed_macs"] / 1e6, "o-", label="Dense executed MACs", color="#E15759")
        axes[1, 0].plot(x, comparison["theoretical_nonzero_macs"] / 1e6, "s--", label="Nonzero MACs (theoretical)", color="#59A14F")
        axes[1, 0].set(title="C · Compute accounting", xlabel="Kernel sparsity (%)", ylabel="MACs (millions)")
        axes[1, 0].legend(frameon=False)
        axes[1, 1].plot(x, comparison["tflite_live_activation_peak_bytes"] / 1024, "o-", label="Live activation", color="#B07AA1")
        axes[1, 1].plot(x, comparison["host_median_ms"], "s--", label="Host median (not ESP32)", color="#F28E2B")
        axes[1, 1].set(title="D · Dense shapes remain unchanged", xlabel="Kernel sparsity (%)", ylabel="KiB or ms")
        axes[1, 1].legend(frameon=False)
        fig.suptitle("Paper-inspired unstructured pruning versus the Fast-80 pruning baseline", fontsize=16, weight="bold", y=1.02)
        figure_path = FIGURE_DIR / "efficiency_dashboard_revised.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.show()
        print(figure_path.relative_to(ROOT))
        """
    ),
    md("## 31 · Load the exact selected-model ESP32 profile when available"),
    code(
        """
        DEVICE_VARIANT = f"prune_unstructured_iterative_{SELECTED_TAG}"
        candidate_device_dir = ROOT / "artifacts/device_profiles" / DEVICE_VARIANT
        candidate_device_summary_path = candidate_device_dir / "device_profile_summary.json"
        baseline_device_summary_path = BASELINE_DEVICE_DIR / "device_profile_summary.json"
        device_profile_available = candidate_device_summary_path.exists() and baseline_device_summary_path.exists()
        device_profile_command = (
            "source scripts/activate_esp_idf.sh && "
            f"scripts/profile_esp32_model.sh {SELECTED_TFLITE_PATH.relative_to(ROOT)} 80 {DEVICE_VARIANT} /dev/cu.usbserial-120"
        )
        if device_profile_available:
            print(f"Physical profile found: {candidate_device_dir.relative_to(ROOT)}")
        else:
            print("Physical profile pending. Run:")
            print(device_profile_command)
        """
    ),
    md("## 32 · Compare complete physical resource evidence"),
    code(
        """
        device_comparison = pd.DataFrame()
        if device_profile_available:
            with baseline_device_summary_path.open(encoding="utf-8") as handle:
                baseline_device = json.load(handle)
            with candidate_device_summary_path.open(encoding="utf-8") as handle:
                candidate_device = json.load(handle)
            device_metrics = [
                ("Model bytes", "model_bytes", "B"),
                ("Dense MACs", "estimated_macs", "MAC"),
                ("Live activation peak", "tflite_live_activation_peak_bytes", "B"),
                ("TFLM arena used", "arena_used_bytes", "B"),
                ("Internal SRAM free", "ready_internal_free_bytes", "B"),
                ("Largest SRAM block", "ready_internal_largest_bytes", "B"),
                ("Mapped PSRAM free", "ready_psram_free_bytes", "B"),
                ("Invoke mean", "mean_us", "µs"),
            ]
            rows = []
            for label, key, unit in device_metrics:
                if key in baseline_device and key in candidate_device:
                    rows.append({"metric": label, "key": key, "unit": unit, "baseline": baseline_device[key], "candidate": candidate_device[key], "candidate_over_baseline": candidate_device[key] / baseline_device[key]})
            pipeline_keys = [
                ("Preprocess median", "pipeline_preprocess_median_ms", "ms"),
                ("Active pipeline median", "pipeline_active_median_ms", "ms"),
                ("Frame period median", "pipeline_period_median_ms", "ms"),
                ("Complete pipeline FPS", "pipeline_fps_median", "fps"),
            ]
            for label, key, unit in pipeline_keys:
                if key in baseline_device and key in candidate_device:
                    rows.append({"metric": label, "key": key, "unit": unit, "baseline": baseline_device[key], "candidate": candidate_device[key], "candidate_over_baseline": candidate_device[key] / baseline_device[key]})
            device_comparison = pd.DataFrame(rows)
            device_comparison.to_csv(REPORT_DIR / "physical_device_comparison.csv", index=False)
            fig, ax = plt.subplots(figsize=(12, 7))
            positions = np.arange(len(device_comparison))
            ratios = device_comparison["candidate_over_baseline"]
            colors_physical = ["#59A14F" if ratio < 0.995 else "#E15759" if ratio > 1.005 else "#9C9C9C" for ratio in ratios]
            bars = ax.barh(device_comparison["metric"], ratios, color=colors_physical)
            ax.axvline(1.0, color="#444444", linewidth=1.3)
            ax.set(title="Physical ESP32-CAM resource ratio: selected candidate / Fast-80", xlabel="Ratio to Fast-80 (1.0 = unchanged)", ylabel="")
            for bar, ratio in zip(bars, ratios):
                ax.text(ratio + 0.01, bar.get_y() + bar.get_height() / 2, f"{ratio:.3f}×", va="center", fontsize=9)
            figure_path = FIGURE_DIR / "physical_esp32_comparison.png"
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.show()
            display(device_comparison.style.hide(axis="index").format({"baseline": "{:,.3f}", "candidate": "{:,.3f}", "candidate_over_baseline": "{:.3f}×"}))
        else:
            display(Markdown("**Physical-device status: pending.** No ESP32 speed or memory claim is made until the exact selected FlatBuffer is captured."))
        """
    ),
    md("## 33 · Write the corrected experiment report"),
    code(
        """
        selected_resource = resources.set_index("export_key").loc[SELECTED_EXPORT_KEY]
        baseline_resource = resources[resources["strategy"] == "baseline"].iloc[0]
        head_rows = layer_sparsity[layer_sparsity["layer"] == "person_probability"].pivot(index="tag", columns="strategy", values="layer_sparsity")
        physical_status = f"Available in `{candidate_device_dir.relative_to(ROOT)}`." if device_profile_available else f"Pending. Run `{device_profile_command}`."
        report_lines = [
            "# Unstructured magnitude pruning report — revised",
            "",
            "## Method contract",
            "",
            "- Fast-80 is the frozen pruning baseline.",
            "- Global one-shot is retained as the negative control.",
            f"- Paper-inspired branch uses {ITERATIVE_ROUNDS} prune/retrain rounds, sensitivity-adjusted magnitude, and Adam {ITERATIVE_LEARNING_RATE:g}.",
            "- This is a Fast-80 adaptation of Han et al., not an exact AlexNet/VGG reproduction.",
            f"- Selected candidate: `{SELECTED_EXPORT_KEY}` ({SELECTED_SPARSITY:.0%}) — {selection_reason}.",
            "- Selection and thresholds use validation only; held-out test is opened after selection.",
            "",
            "## Held-out full-INT8 test",
            "",
            "| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | PR-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Fast-80 pruning baseline | {baseline_test_metrics['threshold']:.3f} | {baseline_test_metrics['accuracy']:.3f} | {baseline_test_metrics['precision']:.3f} | {baseline_test_metrics['recall']:.3f} | {baseline_test_metrics['specificity']:.3f} | {baseline_test_metrics['f1']:.3f} | {baseline_test_metrics['pr_auc']:.3f} |",
            f"| Paper-inspired {SELECTED_TAG} | {selected_test_metrics['threshold']:.3f} | {selected_test_metrics['accuracy']:.3f} | {selected_test_metrics['precision']:.3f} | {selected_test_metrics['recall']:.3f} | {selected_test_metrics['specificity']:.3f} | {selected_test_metrics['f1']:.3f} | {selected_test_metrics['pr_auc']:.3f} |",
            "",
            "## Efficiency interpretation",
            "",
            f"- Kernel sparsity: {selected_resource['kernel_sparsity']:.1%}.",
            f"- Raw TFLite: {int(selected_resource['tflite_bytes']):,} B versus {int(baseline_resource['tflite_bytes']):,} B controlled re-export.",
            f"- Gzip diagnostic: {int(selected_resource['gzip_bytes']):,} B versus {int(baseline_resource['gzip_bytes']):,} B.",
            f"- Bitmap + INT8 values estimate: {int(selected_resource['bitmap_packed_kernel_bytes']):,} B for kernels only.",
            f"- Theoretical nonzero MACs: {int(selected_resource['theoretical_nonzero_macs']):,}.",
            f"- Dense executed MACs remain {int(selected_resource['dense_executed_macs']):,}; activation peak remains {int(selected_resource['tflite_live_activation_peak_bytes']):,} B.",
            "- Sparse acceleration still requires a compatible storage format and custom sparse ESP-NN kernels.",
            "",
            "## Physical ESP32-CAM profile",
            "",
            physical_status,
            "",
            "## Next technique",
            "",
            "Notebook 14 is pattern-based pruning, following the granularity sequence in the pruning plan.",
        ]
        report_path = OUTPUT_DIR / "UNSTRUCTURED_MAGNITUDE_REPORT.md"
        report_path.write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")
        print(report_path.relative_to(ROOT))
        """
    ),
    md("## 34 · Record a machine-readable artifact manifest"),
    code(
        """
        generated_files = sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file())
        manifest = {
            "notebook": "optimization/pruning/13_unstructured_magnitude_pruning.ipynb",
            "technique": "global one-shot control plus sensitivity-adjusted iterative unstructured magnitude pruning",
            "primary_reference": "Han et al. 2015, Learning both Weights and Connections",
            "seed": SEED,
            "sparsity_levels": list(SPARSITY_LEVELS),
            "iterative_rounds": ITERATIVE_ROUNDS,
            "selected_export_key": SELECTED_EXPORT_KEY,
            "selected_sparsity": SELECTED_SPARSITY,
            "selection_reason": selection_reason,
            "physical_device_profile_available": bool(device_profile_available),
            "physical_device_variant": DEVICE_VARIANT,
            "flash_led_enabled_during_profile": False,
            "frozen_input_hashes": current_hashes,
            "selected_float_sha256": sha256_file(SELECTED_FLOAT_PATH),
            "selected_tflite_sha256": sha256_file(SELECTED_TFLITE_PATH),
            "generated_artifacts": [
                {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in generated_files
                if path.name != "artifact_manifest.json" and path.suffix != ".keras"
            ],
        }
        manifest_path = OUTPUT_DIR / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
        print({"selected": SELECTED_EXPORT_KEY, "physical_profile": device_profile_available, "artifacts": len(manifest["generated_artifacts"])})
        """
    ),
    md("## 35 · Findings and hand-off"),
    code(
        """
        device_sentence = "Physical ESP32 evidence is included above." if device_profile_available else "Physical ESP32 capture is still required before a latency or memory conclusion."
        conclusion = (
            f"**Notebook 13 revised: `{SELECTED_EXPORT_KEY}` selected at {SELECTED_SPARSITY:.0%} sparsity.**\\n\\n"
            f"- Held-out INT8 F1: **{selected_test_metrics['f1']:.3f}** versus **{baseline_test_metrics['f1']:.3f}** for the controlled Fast-80 re-export.\\n"
            f"- Recall: **{selected_test_metrics['recall']:.3f}**; PR-AUC: **{selected_test_metrics['pr_auc']:.3f}**.\\n"
            f"- Raw TFLite: **{int(selected_resource['tflite_bytes']):,} B**; dense MACs: **{int(selected_resource['dense_executed_macs']):,}**; live activation: **{int(selected_resource['tflite_live_activation_peak_bytes']):,} B**.\\n"
            f"- Theoretical nonzero MACs: **{int(selected_resource['theoretical_nonzero_macs']):,}**; this is not a dense ESP-NN speed claim.\\n"
            f"- {device_sentence}\\n"
            "- **Next granularity:** Notebook 14, pattern-based pruning."
        )
        display(Markdown(conclusion))
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(cells)} cells")
