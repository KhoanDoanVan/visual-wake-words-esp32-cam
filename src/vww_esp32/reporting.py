from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_model_card(config: dict, dataset_summary: dict, metrics: dict, export_info: dict) -> str:
    threshold = metrics.get("threshold", "not evaluated")
    return f"""# Model card: Visual Wake Words for ESP32-CAM

Generated: {datetime.now(timezone.utc).isoformat()}

## Intended use

Binary, image-level detection of a sufficiently large person in an ESP32-CAM frame. This is a wake-word gate, not a person detector: it does not localize, identify, or track people.

## Data

- Source: {config["data"]["dataset"]} real-world images and instance annotations.
- Positive rule: at least one non-crowd `person` box occupies at least {config["data"]["min_person_area_fraction"]:.1%} of image area.
- Train / validation / test samples: {dataset_summary.get("counts", {})}.
- Splitting: deterministic train/validation partition from COCO train; untouched COCO validation subset as test.

## Model and preprocessing

- Architecture: MobileNetV1-style depthwise-separable CNN, width multiplier {config["model"]["width_multiplier"]}.
- Input: RGB {config["preprocessing"]["image_size"][0]}×{config["preprocessing"]["image_size"][1]}, bilinear resize.
- Export: full integer INT8 TensorFlow Lite; model size {export_info.get("size_bytes", "not exported")} bytes.
- Decision threshold: {threshold} (selected on validation only).

## Held-out test results

{chr(10).join(f"- {key}: {value}" for key, value in metrics.items() if key not in {"confusion_matrix"})}

Confusion matrix (`[[TN, FP], [FN, TP]]`): `{metrics.get("confusion_matrix", "not evaluated")}`.

## Limitations and responsible use

- COCO is not representative of every room, camera position, geography, lighting condition, or ESP32 sensor.
- Small or partially occluded people below the labeling threshold are deliberately negative; this may be surprising in deployment.
- Camera JPEG, lens, resizing, and exposure shift can reduce accuracy. Validate with device-captured data before release.
- Do not use for identity, surveillance decisions, safety-critical automation, or as proof that a space is unoccupied.
- Measure latency and tensor-arena high-water usage on the exact board; desktop TFLite success is necessary but insufficient.
"""


def write_model_card(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
