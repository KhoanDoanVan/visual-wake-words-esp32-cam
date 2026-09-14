from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([np.asarray(y).reshape(-1) for _, y in dataset])
    probabilities = np.asarray(model.predict(dataset, verbose=1)).reshape(-1)
    return labels.astype(int), probabilities


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "f1",
    minimum_recall: float = 0.8,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 2.0,
) -> dict[str, float]:
    thresholds = np.linspace(0.01, 0.99, 197)
    candidates = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        _, fp, fn, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        candidates.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "cost": float(fp * false_positive_cost + fn * false_negative_cost),
            }
        )
    eligible = [item for item in candidates if item["recall"] >= minimum_recall] or candidates
    if objective == "cost":
        return min(eligible, key=lambda item: (item["cost"], -item["f1"]))
    return max(eligible, key=lambda item: (item["f1"], -item["cost"]))


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    return {
        "samples": len(labels),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": matrix.tolist(),
        "positive_prevalence": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(predictions)),
    }


def expected_calibration_error(labels, probabilities, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def calibration_points(labels, probabilities, bins=10):
    observed, predicted = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    return predicted, observed
