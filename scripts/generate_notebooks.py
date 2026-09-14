#!/usr/bin/env python3
"""Generate the committed, source-clean notebook suite."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(title: str, objective: str, cells: list[dict]) -> dict:
    intro = markdown(
        f"# {title}\n\n**Single responsibility:** {objective}\n\n"
        "Run after the preceding numbered notebook unless the inputs already exist. "
        "Every generated artifact is written outside the notebook so this stage is reproducible.\n"
    )
    assembled = [intro, *deepcopy(cells)]
    for index, cell in enumerate(assembled):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": assembled,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = code(
    "from pathlib import Path\n"
    "import sys\n\n"
    "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/base.yaml').exists())\n"
    "if str(ROOT / 'src') not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT / 'src'))\n\n"
    "from vww_esp32.config import load_config, resolve_paths, seed_everything\n\n"
    "config, ROOT = load_config(ROOT / 'configs/base.yaml')\n"
    "paths = resolve_paths(config, ROOT)\n"
    "seed_everything(config['project']['seed'])\n"
    "ROOT\n"
)


NOTEBOOK_DEFINITIONS = {
    "00_project_setup.ipynb": notebook(
        "00 · Project setup and experiment contract",
        "validate the environment, configuration, reproducibility controls, and ESP32 constraints",
        [
            markdown(
                "## Why this design\n\nThe target is image-level **person / no-person** gating, not bounding-box detection. The positive-label rule follows the Visual Wake Words paper: a non-crowd person box must occupy at least 0.5% of image area. The deployment baseline is RGB 96×96, MobileNetV1-style depthwise convolutions, and full INT8 quantization."
            ),
            SETUP,
            code(
                "import platform\nimport numpy as np\nimport pandas as pd\n\nversions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}\ntry:\n    import tensorflow as tf\n    versions['tensorflow'] = tf.__version__\n    versions['accelerators'] = [device.name for device in tf.config.list_physical_devices() if device.device_type != 'CPU']\nexcept ImportError as exc:\n    versions['tensorflow_error'] = str(exc)\nversions"
            ),
            code(
                "required_sections = {'project', 'paths', 'data', 'preprocessing', 'model', 'training', 'evaluation', 'export'}\nmissing = required_sections - config.keys()\nassert not missing, f'Missing config sections: {missing}'\nassert config['preprocessing']['image_size'] == [96, 96]\nassert 0 < config['data']['min_person_area_fraction'] < 1\nconfig"
            ),
            code(
                "disk_target_gb = 8  # conservative budget for the default selected-image experiment\nimport shutil\nfree_gb = shutil.disk_usage(ROOT).free / 1024**3\nprint(f'Free disk: {free_gb:.1f} GiB')\nif free_gb < disk_target_gb:\n    print('WARNING: lower data.max_samples before acquisition or free disk space.')"
            ),
            markdown(
                "## Experiment contract\n\n- Tune architecture and threshold with `train` + `val` only.\n- Open the held-out COCO-derived `test` labels only in notebook 06.\n- Report PR-AUC, recall, specificity, F1, calibration, model bytes, and desktop INT8 parity.\n- Release only after device-captured validation, measured latency, and measured tensor-arena high-water usage."
            ),
        ],
    ),
    "01_data_raw.ipynb": notebook(
        "01 · Raw data acquisition",
        "download and verify official COCO 2017 instance annotations without downloading the full image archive",
        [
            SETUP,
            markdown(
                "## Source and license\n\nThis pipeline uses real MS COCO 2017 photographs. Review [COCO terms of use](https://cocodataset.org/#termsofuse); each image retains its source `license_id`. The annotation archive is about 241 MB. Images are selected and downloaded individually in notebook 03."
            ),
            code(
                "from vww_esp32.data import download_file, sha256_file\n\narchive = paths['raw'] / 'annotations_trainval2017.zip'\narchive = download_file(config['data']['annotation_url'], archive)\n{'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha256_file(archive)}"
            ),
            code(
                "from vww_esp32.data import extract_zip_safely\n\nannotation_dir = paths['raw'] / 'annotations'\nexpected = [annotation_dir / 'instances_train2017.json', annotation_dir / 'instances_val2017.json']\nif not all(path.exists() for path in expected):\n    extract_zip_safely(archive, paths['raw'])\nassert all(path.exists() and path.stat().st_size > 0 for path in expected)\nexpected"
            ),
            code(
                "import json\n\nraw_audit = {}\nfor annotation_path in expected:\n    with annotation_path.open() as handle:\n        payload = json.load(handle)\n    raw_audit[annotation_path.stem] = {\n        'images': len(payload['images']),\n        'annotations': len(payload['annotations']),\n        'categories': len(payload['categories']),\n        'person_category_id': next(c['id'] for c in payload['categories'] if c['name'] == 'person'),\n    }\nraw_audit"
            ),
        ],
    ),
    "02_eda.ipynb": notebook(
        "02 · Exploratory data analysis",
        "audit label construction, image geometry, person-scale distributions, sensitivity, and representative examples",
        [
            SETUP,
            code(
                "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom vww_esp32.data import annotation_frame, load_coco\n\nsns.set_theme(style='whitegrid', context='notebook')\nannotation_dir = paths['raw'] / 'annotations'\nframes = []\nfor source in ('train2017', 'val2017'):\n    frame = annotation_frame(load_coco(annotation_dir / f'instances_{source}.json'))\n    frame['source'] = source\n    frames.append(frame)\neda = pd.concat(frames, ignore_index=True)\nthreshold = config['data']['min_person_area_fraction']\neda['label'] = (eda['max_person_area_fraction'] >= threshold).astype(int)\neda.head()"
            ),
            markdown(
                "## Dataset shape and class prior\n\nThe 0.5% rule intentionally maps images containing only tiny people to the negative class. Inspect both raw person presence and the deployable VWW label."
            ),
            code(
                "summary = eda.groupby('source').agg(images=('image_id', 'size'), any_person=('person_count', lambda x: (x > 0).mean()), vww_positive=('label', 'mean'), median_width=('width', 'median'), median_height=('height', 'median'))\nsummary.style.format({'any_person': '{:.1%}', 'vww_positive': '{:.1%}'})"
            ),
            code(
                "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.countplot(data=eda, x='source', hue='label', ax=axes[0], palette='Set2')\naxes[0].set(title='VWW class counts', ylabel='images')\naxes[0].legend(title='label', labels=['no person', 'person'])\nsns.histplot(data=eda, x='aspect_ratio', hue='source', bins=60, element='step', stat='density', common_norm=False, ax=axes[1])\naxes[1].axvline(4/3, color='black', linestyle='--', linewidth=1, label='4:3 sensor')\naxes[1].set(xlim=(0.3, 3.0), title='Image aspect ratio (clipped view)')\nplt.tight_layout()"
            ),
            markdown("## Person-scale distribution and threshold sensitivity"),
            code(
                "person_images = eda[eda['max_person_area_fraction'] > 0].copy()\nfig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.ecdfplot(data=person_images, x='max_person_area_fraction', hue='source', ax=axes[0])\naxes[0].axvline(threshold, color='red', linestyle='--', label=f'{threshold:.0%} rule')\naxes[0].set(xlim=(0, 0.5), xlabel='largest person box / image area', title='Person scale ECDF')\nthresholds = np.linspace(0, 0.2, 41)\nsensitivity = pd.DataFrame({'threshold': thresholds, 'positive_rate': [(eda.max_person_area_fraction >= value).mean() for value in thresholds]})\nsns.lineplot(data=sensitivity, x='threshold', y='positive_rate', ax=axes[1])\naxes[1].axvline(threshold, color='red', linestyle='--')\naxes[1].set(title='Label prevalence sensitivity', ylabel='positive image fraction')\nplt.tight_layout()"
            ),
            markdown(
                "## Qualitative label audit\n\nA stratified grid makes the 0.5% boundary tangible and can expose surprising negatives before training."
            ),
            code(
                "from PIL import Image\nfrom vww_esp32.data import coco_download_url, download_file\n\nexamples = pd.concat([group.sample(n=4, random_state=42) for _, group in eda.groupby('label')])\nfigure, axes = plt.subplots(2, 4, figsize=(14, 7))\nfor axis, row in zip(axes.flat, examples.itertuples()):\n    url = coco_download_url(row.coco_url)\n    local = paths['raw'] / 'eda_samples' / row.file_name\n    download_file(url, local)\n    axis.imshow(Image.open(local).convert('RGB'))\n    axis.set_title(f\"label={row.label}; max box={row.max_person_area_fraction:.2%}\")\n    axis.axis('off')\nplt.tight_layout()"
            ),
            markdown(
                "## Sampling-bias checks\n\nResolution, aspect ratio, license mix, person count, and scale can become shortcuts. These tables identify strata to revisit during error analysis."
            ),
            code(
                "pd.crosstab(pd.cut(eda.aspect_ratio, [0, .8, 1.2, 1.6, 10]), eda.label, normalize='index').rename(columns={0: 'negative_rate', 1: 'positive_rate'}).style.format('{:.1%}')"
            ),
            code(
                "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.boxenplot(data=eda.sample(min(20000, len(eda)), random_state=42), x='label', y='width', ax=axes[0])\naxes[0].set(ylim=(0, 1000), title='Width by label (display clipped)')\nsns.histplot(data=person_images, x='person_count', bins=range(1, 16), discrete=True, ax=axes[1])\naxes[1].set(xlim=(0, 15), title='People per annotated image')\nplt.tight_layout()"
            ),
            markdown(
                "## Findings to carry forward\n\nWrite observations here after execution. At minimum address class prior, the fraction of ambiguous small-person images, geometry shift from resizing, and whether the sampled training balance differs from deployment prevalence."
            ),
        ],
    ),
    "03_data_extraction.ipynb": notebook(
        "03 · Label extraction and selected-image download",
        "create deterministic train/validation/test manifests and acquire only the real COCO images used",
        [
            SETUP,
            code(
                "from vww_esp32.data import build_manifest\n\nannotation_dir = paths['raw'] / 'annotations'\nmanifest = build_manifest(\n    annotation_dir / 'instances_train2017.json',\n    annotation_dir / 'instances_val2017.json',\n    paths['raw'] / 'images',\n    min_area_fraction=config['data']['min_person_area_fraction'],\n    validation_fraction=config['data']['validation_fraction'],\n    max_samples=config['data']['max_samples'],\n    balance_training_classes=config['data']['balance_training_classes'],\n    seed=config['project']['seed'],\n)\nmanifest.groupby(['split', 'label']).size().rename('images').to_frame()"
            ),
            markdown(
                "The training subset is balanced to improve optimization. Validation and test preserve their sampled COCO prevalence so operating metrics remain interpretable."
            ),
            code(
                "from vww_esp32.data import download_manifest_images\n\nmanifest = download_manifest_images(manifest, workers=config['data']['download_workers'], timeout=config['data']['download_timeout_seconds'])\nfailures = manifest.attrs.get('failures', [])\nprint(f'Downloaded: {manifest.downloaded.sum():,}/{len(manifest):,}; failures: {len(failures):,}')\nfailures[:5]"
            ),
            code(
                "interim_manifest = paths['interim'] / 'manifest.csv'\nmanifest.to_csv(interim_manifest, index=False)\nassert manifest.image_id.is_unique\nassert set(manifest.split) == {'train', 'val', 'test'}\nassert set(manifest.label) <= {0, 1}\ninterim_manifest"
            ),
        ],
    ),
    "04_data_preprocessing.ipynb": notebook(
        "04 · Data validation and preprocessing",
        "reject corrupt inputs, audit duplicates and brightness, then freeze the processed manifest and tf.data contract",
        [
            SETUP,
            code(
                "import pandas as pd\nfrom vww_esp32.data import validate_images\n\nmanifest = pd.read_csv(paths['interim'] / 'manifest.csv')\nvalidated = validate_images(manifest)\nvalidated.groupby(['split', 'valid']).size().rename('images').to_frame()"
            ),
            code(
                "from vww_esp32.data import sha256_file\n\nvalid = validated[validated.valid].copy()\nvalid['sha256'] = valid.image_path.map(sha256_file)\nduplicates = valid[valid.duplicated('sha256', keep=False)].sort_values('sha256')\nprint(f'Exact duplicate rows: {len(duplicates):,}')\nduplicates[['split', 'image_id', 'sha256']].head(10)"
            ),
            markdown(
                "Cross-split exact duplicates are a leakage risk. COCO image IDs should already be unique; the content hash is a defensive check."
            ),
            code(
                "cross_split = duplicates.groupby('sha256').split.nunique() if len(duplicates) else pd.Series(dtype=int)\nassert not (cross_split > 1).any(), 'Exact image content leaks across splits'"
            ),
            code(
                "import numpy as np\nfrom PIL import Image, ImageStat\n\ndef image_quality(path):\n    with Image.open(path) as image:\n        gray = image.convert('L').resize((32, 32))\n        stat = ImageStat.Stat(gray)\n        return pd.Series({'brightness': stat.mean[0], 'contrast': stat.stddev[0]})\n\nquality = valid.image_path.map(image_quality)\nquality = pd.DataFrame(quality.tolist(), index=valid.index)\nvalid = valid.join(quality)\nvalid.groupby(['split', 'label'])[['brightness', 'contrast']].agg(['mean', 'std']).round(1)"
            ),
            code(
                "import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.kdeplot(data=valid, x='brightness', hue='label', common_norm=False, ax=axes[0])\nsns.kdeplot(data=valid, x='contrast', hue='label', common_norm=False, ax=axes[1])\naxes[0].set_title('Brightness by class')\naxes[1].set_title('Contrast by class')\nplt.tight_layout()"
            ),
            code(
                "processed_manifest = paths['processed'] / 'manifest.csv'\nvalid.drop(columns=['valid', 'validation_error'], errors='ignore').to_csv(processed_manifest, index=False)\nprint(valid.groupby(['split', 'label']).size())\nprint(processed_manifest)"
            ),
            code(
                "from vww_esp32.preprocessing import make_dataset, split_manifest\n\nsplits = split_manifest(valid)\ntrain_ds = make_dataset(splits['train'], tuple(config['preprocessing']['image_size']), config['preprocessing']['batch_size'], training=True, seed=config['project']['seed'])\nimages, labels = next(iter(train_ds))\nassert images.shape[1:] == (96, 96, 3)\nassert images.dtype.name == 'float32'\n{'image_batch': images.shape, 'labels': labels.shape, 'pixel_range': (float(images.numpy().min()), float(images.numpy().max()))}"
            ),
        ],
    ),
    "05_training.ipynb": notebook(
        "05 · Model training",
        "train a compact MobileNetV1-style classifier and persist the best checkpoint plus learning curves",
        [
            SETUP,
            code(
                "import json\nimport pandas as pd\nimport tensorflow as tf\nfrom vww_esp32.preprocessing import make_dataset, split_manifest\n\nmanifest = pd.read_csv(paths['processed'] / 'manifest.csv')\nsplits = split_manifest(manifest)\nimage_size = tuple(config['preprocessing']['image_size'])\nbatch_size = config['preprocessing']['batch_size']\ntrain_ds = make_dataset(splits['train'], image_size, batch_size, training=True, seed=config['project']['seed'], cache=config['preprocessing']['cache'])\nval_ds = make_dataset(splits['val'], image_size, batch_size, training=False, cache=config['preprocessing']['cache'])\nprint({name: len(frame) for name, frame in splits.items()})"
            ),
            code(
                "from vww_esp32.modeling import build_tiny_mobilenet_v1, compile_model\n\nmodel = build_tiny_mobilenet_v1(input_shape=(*image_size, 3), alpha=config['model']['width_multiplier'], dropout=config['model']['dropout'], l2=config['model']['l2'])\nmodel = compile_model(model, config['training']['learning_rate'], config['training']['label_smoothing'])\nmodel.summary()\nprint(f'Parameters: {model.count_params():,}')"
            ),
            code(
                "from vww_esp32.modeling import training_callbacks\n\ncallbacks = training_callbacks(paths['artifacts'], monitor=config['training']['monitor'], patience=config['training']['early_stopping_patience'], lr_patience=config['training']['reduce_lr_patience'])\nhistory = model.fit(train_ds, validation_data=val_ds, epochs=config['training']['epochs'], callbacks=callbacks)"
            ),
            code(
                "import matplotlib.pyplot as plt\nimport pandas as pd\n\nhistory_frame = pd.DataFrame(history.history)\nmetrics = [('loss', 'val_loss'), ('accuracy', 'val_accuracy'), ('pr_auc', 'val_pr_auc'), ('recall', 'val_recall')]\nfig, axes = plt.subplots(2, 2, figsize=(13, 9))\nfor axis, pair in zip(axes.flat, metrics):\n    history_frame[list(pair)].plot(ax=axis, title=pair[0])\n    axis.set_xlabel('epoch')\nplt.tight_layout()\nfigure_path = paths['artifacts'] / 'figures' / 'training_curves.png'\nfig.savefig(figure_path, dpi=160, bbox_inches='tight')"
            ),
            code(
                "final_model = paths['artifacts'] / 'models' / 'final.keras'\nmodel.save(final_model)\nmetadata = {'seed': config['project']['seed'], 'tensorflow': tf.__version__, 'epochs_completed': len(history_frame), 'best_val_pr_auc': float(history_frame.val_pr_auc.max()), 'parameters': model.count_params()}\n(paths['artifacts'] / 'reports' / 'training_metadata.json').write_text(json.dumps(metadata, indent=2))\nmetadata"
            ),
        ],
    ),
    "06_evaluation.ipynb": notebook(
        "06 · Evaluation and error analysis",
        "choose the threshold on validation, lock it, evaluate once on test, and examine calibration and failure slices",
        [
            SETUP,
            code(
                "import pandas as pd\nimport tensorflow as tf\nfrom vww_esp32.preprocessing import make_dataset, split_manifest\nfrom vww_esp32.evaluation import collect_predictions, select_threshold, classification_metrics, expected_calibration_error, save_metrics\n\nmanifest = pd.read_csv(paths['processed'] / 'manifest.csv')\nsplits = split_manifest(manifest)\nimage_size = tuple(config['preprocessing']['image_size'])\nmodel = tf.keras.models.load_model(paths['artifacts'] / 'checkpoints' / 'best.keras')\nval_ds = make_dataset(splits['val'], image_size, config['preprocessing']['batch_size'])\ny_val, p_val = collect_predictions(model, val_ds)"
            ),
            code(
                "choice = select_threshold(y_val, p_val, objective=config['evaluation']['threshold_objective'], minimum_recall=config['evaluation']['minimum_recall'], false_positive_cost=config['evaluation']['false_positive_cost'], false_negative_cost=config['evaluation']['false_negative_cost'])\nchoice"
            ),
            markdown(
                "The threshold is now locked. The following cell is the first use of the held-out test labels."
            ),
            code(
                "test_ds = make_dataset(splits['test'], image_size, config['preprocessing']['batch_size'])\ny_test, p_test = collect_predictions(model, test_ds)\ntest_metrics = classification_metrics(y_test, p_test, choice['threshold'])\ntest_metrics['expected_calibration_error_10_bins'] = expected_calibration_error(y_test, p_test, bins=10)\nsave_metrics(test_metrics, paths['artifacts'] / 'reports' / 'test_metrics.json')\ntest_metrics"
            ),
            code(
                "import matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay\nfrom vww_esp32.evaluation import calibration_points\n\nfig, axes = plt.subplots(2, 2, figsize=(12, 10))\nConfusionMatrixDisplay.from_predictions(y_test, p_test >= choice['threshold'], display_labels=['no person', 'person'], cmap='Blues', ax=axes[0, 0], colorbar=False)\nRocCurveDisplay.from_predictions(y_test, p_test, ax=axes[0, 1])\nPrecisionRecallDisplay.from_predictions(y_test, p_test, ax=axes[1, 0])\npredicted, observed = calibration_points(y_test, p_test)\naxes[1, 1].plot([0, 1], [0, 1], '--', color='gray')\naxes[1, 1].plot(predicted, observed, marker='o')\naxes[1, 1].set(xlabel='mean predicted probability', ylabel='observed fraction', title='Calibration')\nplt.tight_layout()\nfig.savefig(paths['artifacts'] / 'figures' / 'test_evaluation.png', dpi=160, bbox_inches='tight')"
            ),
            code(
                "predictions = splits['test'].copy()\npredictions['probability'] = p_test\npredictions['prediction'] = (p_test >= choice['threshold']).astype(int)\npredictions['correct'] = predictions.label == predictions.prediction\npredictions.to_csv(paths['artifacts'] / 'reports' / 'test_predictions.csv', index=False)\nslices = predictions.assign(brightness_bin=pd.qcut(predictions.brightness, 4, duplicates='drop'), scale_bin=pd.cut(predictions.max_person_area_fraction, [-1, .01, .05, .15, 1])).groupby(['brightness_bin', 'label'], observed=True).correct.agg(['count', 'mean'])\nslices"
            ),
            code(
                "import numpy as np\n\nerrors = predictions[~predictions.correct].copy()\nerrors['confidence'] = np.where(errors.prediction == 1, errors.probability, 1-errors.probability)\nerrors.sort_values('confidence', ascending=False)[['image_path', 'label', 'probability', 'max_person_area_fraction', 'brightness', 'contrast']].head(20)"
            ),
        ],
    ),
    "07_model_profiling.ipynb": notebook(
        "07 · Pre-optimization model profiling",
        "profile the frozen trained model's layers, sparsity, compute, and memory before any pruning or quantization",
        [
            markdown(
                "## Scope and interpretation\n\nThis notebook is deliberately **read-only with respect to the model**. It profiles the best trained float32 checkpoint before manual optimization. Exact-zero sparsity measures existing zeros; it does not imply that small nonzero weights can be removed safely. Memory figures are structural graph estimates, not TensorFlow process RSS or a later TFLite Micro tensor-arena measurement."
            ),
            SETUP,
            code(
                "import json\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nimport tensorflow as tf\nfrom IPython.display import Markdown, display\nfrom matplotlib.ticker import PercentFormatter\n\nfrom vww_esp32.profiling import human_bytes, profile_model\n\nsns.set_theme(style='whitegrid', context='notebook', font_scale=0.95)\nCOLORS = {'parameters': '#355C7D', 'macs': '#F67280', 'activation': '#6C5B7B', 'int8': '#2A9D8F'}\ncheckpoint = paths['artifacts'] / 'checkpoints' / 'best.keras'\nassert checkpoint.exists(), f'Missing trained checkpoint: {checkpoint}'\nmodel = tf.keras.models.load_model(checkpoint, compile=False)\nprint(f'Checkpoint: {checkpoint}')\nprint(f'Checkpoint file: {human_bytes(checkpoint.stat().st_size)}')\nmodel.summary()"
            ),
            markdown("## 1. Executive resource summary"),
            code(
                "layer_profile, memory_trace, profile_summary = profile_model(\n    model,\n    batch_size=1,\n    training_batch_size=config['preprocessing']['batch_size'],\n    near_zero_threshold=1e-6,\n)\n\nsummary_view = pd.DataFrame({\n    'Metric': [\n        'Layers', 'Total parameters', 'Trainable parameters', 'Non-trainable parameters',\n        'Float32 weight memory', 'Estimated MACs / image', 'Global weight sparsity',\n        'Kernel sparsity', 'Kernel near-zero fraction (|w| ≤ 1e-6)',\n        'Peak live float32 activations (batch 1)', 'Hypothetical INT8 activation peak',\n        'Float32 weights + peak activations (combined)',\n        f\"Training structural estimate (batch {profile_summary['training_batch_size_for_estimate']})\",\n    ],\n    'Value': [\n        f\"{profile_summary['layers']:,}\",\n        f\"{profile_summary['parameters']:,}\",\n        f\"{profile_summary['trainable_parameters']:,}\",\n        f\"{profile_summary['non_trainable_parameters']:,}\",\n        human_bytes(profile_summary['float32_weight_bytes']),\n        f\"{profile_summary['estimated_macs_batch1'] / 1e6:.3f} M\",\n        f\"{profile_summary['global_weight_sparsity']:.3%}\",\n        f\"{profile_summary['global_kernel_sparsity']:.3%}\",\n        f\"{profile_summary['global_kernel_near_zero_fraction']:.4%}\",\n        human_bytes(profile_summary['peak_live_activation_float32_bytes_batch1']),\n        human_bytes(profile_summary['peak_live_activation_int8_bytes_batch1_hypothetical']),\n        human_bytes(profile_summary['float32_inference_structural_bytes']),\n        human_bytes(profile_summary['training_structural_bytes_estimate']),\n    ],\n})\ndisplay(summary_view.style.hide(axis='index').set_properties(**{'text-align': 'left'}))\nprint('Peak activation occurs at:', profile_summary['peak_live_activation_layer'])\nprint('Caveat:', profile_summary['memory_estimate_scope'])"
            ),
            markdown("## 2. Layer-by-layer inventory"),
            code(
                "inventory_columns = [\n    'layer_index', 'layer_name', 'layer_type', 'stage', 'input_shape', 'output_shape',\n    'parameters', 'trainable_parameters', 'non_trainable_parameters', 'weight_sparsity',\n    'near_zero_fraction', 'output_float32_bytes_batch1', 'estimated_macs_batch1',\n]\nwith pd.option_context('display.max_rows', 100, 'display.max_colwidth', 40):\n    display(\n        layer_profile[inventory_columns].style\n        .format({\n            'parameters': '{:,.0f}',\n            'trainable_parameters': '{:,.0f}',\n            'non_trainable_parameters': '{:,.0f}',\n            'weight_sparsity': '{:.2%}',\n            'near_zero_fraction': '{:.2%}',\n            'output_float32_bytes_batch1': '{:,.0f}',\n            'estimated_macs_batch1': '{:,.0f}',\n        }, na_rep='—')\n        .background_gradient(subset=['parameters'], cmap='Blues')\n        .background_gradient(subset=['estimated_macs_batch1'], cmap='Reds')\n        .background_gradient(subset=['output_float32_bytes_batch1'], cmap='Purples')\n    )"
            ),
            code(
                "stage_profile = (\n    layer_profile.groupby('stage', sort=False)\n    .agg(\n        layers=('layer_name', 'count'),\n        parameters=('parameters', 'sum'),\n        weight_bytes=('weight_bytes', 'sum'),\n        output_float32_bytes=('output_float32_bytes_batch1', 'sum'),\n        estimated_macs=('estimated_macs_batch1', 'sum'),\n    )\n    .assign(\n        parameter_share=lambda frame: frame.parameters / frame.parameters.sum(),\n        mac_share=lambda frame: frame.estimated_macs / frame.estimated_macs.sum(),\n    )\n)\ndisplay(stage_profile.style.format({\n    'parameters': '{:,.0f}', 'weight_bytes': '{:,.0f}',\n    'output_float32_bytes': '{:,.0f}', 'estimated_macs': '{:,.0f}',\n    'parameter_share': '{:.1%}', 'mac_share': '{:.1%}',\n}).background_gradient(subset=['parameter_share', 'mac_share'], cmap='YlOrRd'))"
            ),
            markdown("## 3. Resource concentration dashboard"),
            code(
                "fig, axes = plt.subplots(2, 2, figsize=(16, 11))\nfig.suptitle('Pre-optimization resource concentration', fontsize=18, fontweight='bold', x=0.06, ha='left')\nfig.text(0.06, 0.94, 'Frozen float32 checkpoint · batch 1 · structural estimates', color='#555555')\n\ndef horizontal_top(axis, column, color, title, unit_scale=1, suffix=''):\n    top = layer_profile.nlargest(12, column).sort_values(column)\n    values = top[column] / unit_scale\n    axis.barh(top.layer_name, values, color=color, alpha=0.88)\n    axis.set_title(title, loc='left', fontweight='bold')\n    axis.set_xlabel(suffix)\n    axis.grid(axis='x', alpha=0.25)\n    axis.grid(axis='y', visible=False)\n\nhorizontal_top(axes[0, 0], 'parameters', COLORS['parameters'], 'A · Parameter-heavy layers', 1e3, 'thousand parameters')\nhorizontal_top(axes[0, 1], 'estimated_macs_batch1', COLORS['macs'], 'B · Compute-heavy layers', 1e6, 'million MACs / image')\nhorizontal_top(axes[1, 0], 'output_float32_bytes_batch1', COLORS['activation'], 'C · Largest layer outputs', 1024, 'KiB, float32 batch 1')\n\npareto = layer_profile[layer_profile.parameters > 0].sort_values('parameters', ascending=False).copy()\npareto['cumulative'] = pareto.parameters.cumsum() / pareto.parameters.sum()\naxes[1, 1].bar(range(len(pareto)), pareto.parameters / 1e3, color=COLORS['parameters'], alpha=0.75)\npareto_axis = axes[1, 1].twinx()\npareto_axis.plot(range(len(pareto)), pareto.cumulative, color='#E76F51', marker='o', markersize=3)\npareto_axis.axhline(0.8, color='#555555', linestyle='--', linewidth=1)\npareto_axis.yaxis.set_major_formatter(PercentFormatter(1))\npareto_axis.set_ylim(0, 1.05)\naxes[1, 1].set(title='D · Parameter Pareto', xlabel='layers ranked by parameter count', ylabel='thousand parameters')\npareto_axis.set_ylabel('cumulative parameter share')\naxes[1, 1].grid(axis='x', visible=False)\n\nplt.tight_layout(rect=(0, 0, 1, 0.92))\noverview_path = paths['artifacts'] / 'figures' / 'pre_optimization_resource_dashboard.png'\nfig.savefig(overview_path, dpi=180, bbox_inches='tight')\nplt.show()"
            ),
            markdown("## 4. Architecture shape and optimization opportunity map"),
            code(
                "spatial = layer_profile.dropna(subset=['output_height', 'output_channels']).copy()\nspatial['activation_kib'] = spatial.output_float32_bytes_batch1 / 1024\ntype_palette = {\n    'InputLayer': '#ADB5BD', 'Conv2D': '#355C7D', 'DepthwiseConv2D': '#2A9D8F',\n    'BatchNormalization': '#E9C46A', 'ReLU': '#F4A261', 'Dense': '#D62828',\n    'Rescaling': '#8D99AE', 'RandomFlip': '#8D99AE', 'RandomTranslation': '#8D99AE',\n}\nfig, axes = plt.subplots(1, 2, figsize=(17, 6.5))\nfig.suptitle('Architecture topology and optimization opportunity', fontsize=18, fontweight='bold', x=0.05, ha='left')\n\nsns.scatterplot(\n    data=spatial, x='layer_index', y='output_height', size='output_channels', hue='layer_type',\n    sizes=(40, 500), palette=type_palette, alpha=0.82, edgecolor='white', linewidth=0.7, ax=axes[0],\n)\naxes[0].plot(spatial.layer_index, spatial.output_height, color='#777777', alpha=0.35, zorder=0)\naxes[0].set_yscale('log', base=2)\naxes[0].set_yticks([3, 6, 12, 24, 48, 96], labels=[3, 6, 12, 24, 48, 96])\naxes[0].set(title='A · Feature-map journey', xlabel='layer index', ylabel='spatial height / width')\naxes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, title='type / channels')\nchanges = spatial[spatial.output_height.ne(spatial.output_height.shift())]\nfor row in changes.itertuples():\n    axes[0].annotate(f'{row.layer_name}\\n{int(row.output_height)}²×{int(row.output_channels)}',\n                     (row.layer_index, row.output_height), xytext=(5, 7), textcoords='offset points', fontsize=7)\n\nresource_layers = layer_profile[layer_profile.estimated_macs_batch1 > 0].copy()\nresource_layers['activation_kib'] = resource_layers.output_float32_bytes_batch1 / 1024\nsns.scatterplot(\n    data=resource_layers, x='estimated_macs_batch1_share', y='parameters_share',\n    size='activation_kib', hue='layer_type', sizes=(70, 700), palette=type_palette,\n    alpha=0.82, edgecolor='white', linewidth=0.8, ax=axes[1],\n)\naxes[1].xaxis.set_major_formatter(PercentFormatter(1))\naxes[1].yaxis.set_major_formatter(PercentFormatter(1))\naxes[1].set(title='B · Compute vs parameter concentration', xlabel='share of total MACs', ylabel='share of parameters')\nlabel_indices = set(resource_layers.nlargest(5, 'estimated_macs_batch1').index) | set(resource_layers.nlargest(5, 'parameters').index)\nfor index in label_indices:\n    row = resource_layers.loc[index]\n    axes[1].annotate(row.layer_name, (row.estimated_macs_batch1_share, row.parameters_share),\n                     xytext=(5, 5), textcoords='offset points', fontsize=8)\naxes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, title='type / activation KiB')\nplt.tight_layout(rect=(0, 0, 1, 0.92))\nopportunity_path = paths['artifacts'] / 'figures' / 'pre_optimization_opportunity_map.png'\nfig.savefig(opportunity_path, dpi=180, bbox_inches='tight')\nplt.show()"
            ),
            markdown("## 5. Existing sparsity and weight-magnitude evidence"),
            code(
                "rng = np.random.default_rng(config['project']['seed'])\nmagnitude_rows = []\nfor layer in model.layers:\n    for variable in layer.trainable_weights:\n        if len(variable.shape) < 2 or 'kernel' not in variable.name.lower():\n            continue\n        values = np.abs(variable.numpy()).reshape(-1)\n        if len(values) > 2500:\n            values = rng.choice(values, size=2500, replace=False)\n        magnitude_rows.extend({\n            'layer_name': layer.name, 'layer_type': layer.__class__.__name__,\n            'log10_abs_weight': np.log10(float(value) + 1e-12),\n        } for value in values)\nweight_magnitudes = pd.DataFrame(magnitude_rows)\n\nfig, axes = plt.subplots(1, 2, figsize=(16, 6))\nfig.suptitle('Sparsity is measured, not assumed', fontsize=18, fontweight='bold', x=0.06, ha='left')\nsns.ecdfplot(data=weight_magnitudes, x='log10_abs_weight', hue='layer_type', palette=type_palette, ax=axes[0])\naxes[0].axvline(-6, color='#E76F51', linestyle='--', label='near-zero threshold')\naxes[0].set(title='A · Kernel magnitude distribution', xlabel='log10(|weight|)', ylabel='cumulative fraction')\n\nkernel_layers = layer_profile.dropna(subset=['kernel_sparsity']).copy()\nnear_zero_top = kernel_layers.nlargest(12, 'near_zero_fraction').sort_values('near_zero_fraction')\naxes[1].barh(near_zero_top.layer_name, near_zero_top.near_zero_fraction, color='#2A9D8F', alpha=0.85)\naxes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=3))\naxes[1].set(\n    title='B · Layers with the largest near-zero fraction',\n    xlabel=f\"fraction with |weight| ≤ {profile_summary['near_zero_threshold']:.0e}\",\n    ylabel='',\n)\naxes[1].grid(axis='y', visible=False)\naxes[1].text(\n    0.99, 0.04, f\"global exact-zero kernel sparsity: {profile_summary['global_kernel_sparsity']:.3%}\",\n    transform=axes[1].transAxes, ha='right', color='#555555', fontsize=9,\n)\nplt.tight_layout(rect=(0, 0, 1, 0.91))\nsparsity_path = paths['artifacts'] / 'figures' / 'pre_optimization_sparsity.png'\nfig.savefig(sparsity_path, dpi=180, bbox_inches='tight')\nplt.show()\n\nprint(f\"Exact-zero sparsity across kernels: {profile_summary['global_kernel_sparsity']:.4%}\")\nprint(f\"Near-zero kernel fraction: {profile_summary['global_kernel_near_zero_fraction']:.4%}\")\nprint('Interpretation: dense nonzero kernels do not become smaller without pruning/retraining and sparse-runtime support.')"
            ),
            markdown("## 6. Peak activation liveness and memory scenarios"),
            code(
                "peak_row = memory_trace.loc[memory_trace.live_float32_bytes.idxmax()]\nmemory_table = pd.DataFrame({\n    'Scenario': [\n        'Float32 weights', 'Peak live float32 activations, batch 1',\n        'Float32 weights + peak activations (combined)', 'Hypothetical INT8 live activations, batch 1',\n        f\"Float32 training structural estimate, batch {profile_summary['training_batch_size_for_estimate']}\",\n    ],\n    'Bytes': [\n        profile_summary['float32_weight_bytes'],\n        profile_summary['peak_live_activation_float32_bytes_batch1'],\n        profile_summary['float32_inference_structural_bytes'],\n        profile_summary['peak_live_activation_int8_bytes_batch1_hypothetical'],\n        profile_summary['training_structural_bytes_estimate'],\n    ],\n})\nmemory_table['Readable'] = memory_table.Bytes.map(human_bytes)\ndisplay(memory_table[['Scenario', 'Readable']].style.hide(axis='index'))\n\nfig, axes = plt.subplots(1, 2, figsize=(17, 6))\nfig.suptitle('Memory pressure before optimization', fontsize=18, fontweight='bold', x=0.05, ha='left')\naxes[0].plot(memory_trace.layer_index, memory_trace.live_float32_bytes / 1024,\n             color=COLORS['activation'], linewidth=2.4, label='float32')\naxes[0].fill_between(memory_trace.layer_index, memory_trace.live_float32_bytes / 1024,\n                     color=COLORS['activation'], alpha=0.16)\naxes[0].plot(memory_trace.layer_index, memory_trace.live_int8_bytes_hypothetical / 1024,\n             color=COLORS['int8'], linewidth=2, linestyle='--', label='hypothetical INT8')\naxes[0].scatter([peak_row.layer_index], [peak_row.live_float32_bytes / 1024], color='#E76F51', zorder=5)\naxes[0].annotate(f\"peak: {peak_row.layer_name}\\n{human_bytes(peak_row.live_float32_bytes)}\",\n                 (peak_row.layer_index, peak_row.live_float32_bytes / 1024),\n                 xytext=(12, -38), textcoords='offset points', arrowprops={'arrowstyle': '->'}, fontsize=9)\naxes[0].set(title='A · Graph-liveness trace', xlabel='layer index', ylabel='live activation KiB')\naxes[0].legend()\n\nplot_memory = memory_table.copy()\nplot_memory['MiB'] = plot_memory.Bytes / 1024**2\naxes[1].barh(plot_memory.Scenario, plot_memory.MiB, color=['#355C7D', '#6C5B7B', '#457B9D', '#2A9D8F', '#E76F51'])\naxes[1].set_xscale('log')\naxes[1].set(title='B · Structural memory scenarios', xlabel='MiB (log scale)')\naxes[1].grid(axis='y', visible=False)\nfor index, row in plot_memory.iterrows():\n    axes[1].text(row.MiB * 1.05, index, row.Readable, va='center', fontsize=9)\nplt.tight_layout(rect=(0, 0, 1, 0.91))\nmemory_path = paths['artifacts'] / 'figures' / 'pre_optimization_memory.png'\nfig.savefig(memory_path, dpi=180, bbox_inches='tight')\nplt.show()"
            ),
            markdown("## 7. Evidence-led optimization priorities"),
            code(
                "top_parameter = layer_profile.nlargest(1, 'parameters').iloc[0]\ntop_compute = layer_profile.nlargest(1, 'estimated_macs_batch1').iloc[0]\ntop_activation = layer_profile.nlargest(1, 'output_float32_bytes_batch1').iloc[0]\nparameter_layers_for_80 = int((layer_profile.sort_values('parameters', ascending=False).parameters.cumsum() / layer_profile.parameters.sum() < .8).sum() + 1)\n\ndisplay(Markdown(f\"\"\"\n### Baseline diagnosis\n\n- **Parameter bottleneck:** `{top_parameter.layer_name}` contains {top_parameter.parameters:,} parameters ({top_parameter.parameters_share:.1%} of the model).\n- **Compute bottleneck:** `{top_compute.layer_name}` performs about {top_compute.estimated_macs_batch1 / 1e6:.3f} M MACs ({top_compute.estimated_macs_batch1_share:.1%} of total).\n- **Activation bottleneck:** `{top_activation.layer_name}` emits {human_bytes(top_activation.output_float32_bytes_batch1)} at batch 1. Graph liveness peaks near `{profile_summary['peak_live_activation_layer']}`.\n- **Concentration:** {parameter_layers_for_80} layers account for roughly 80% of parameters.\n- **Natural sparsity:** kernel exact-zero sparsity is {profile_summary['global_kernel_sparsity']:.3%}; the checkpoint is effectively dense.\n\n### Recommended experiment order\n\n1. Reduce channels in the parameter-heavy 1×1 pointwise convolutions and retrain; this is structured and deployable.\n2. Test a smaller input resolution or earlier downsampling against recall for small people; this targets early activation memory and MACs.\n3. Fold BatchNorm for inference, then compare numerically before any quantization.\n4. If pruning is tested, prefer channel/filter pruning with fine-tuning; unstructured zeros only help when the target runtime has sparse kernels.\n5. Quantize only after the float32 architecture frontier is chosen, using representative ESP32-camera-domain frames.\n\nNo optimization is applied in this notebook. Each future variant should be compared against this frozen baseline.\n\"\"\"))"
            ),
            markdown("## 8. Persist the profiling baseline"),
            code(
                "report_dir = paths['artifacts'] / 'reports'\nfigure_dir = paths['artifacts'] / 'figures'\nreport_dir.mkdir(parents=True, exist_ok=True)\nfigure_dir.mkdir(parents=True, exist_ok=True)\nprofile_summary.update({\n    'checkpoint': str(checkpoint),\n    'checkpoint_bytes': checkpoint.stat().st_size,\n    'profile_status': 'pre_optimization_float32_baseline',\n})\nlayer_path = report_dir / 'model_layer_profile.csv'\nliveness_path = report_dir / 'model_memory_liveness.csv'\nsummary_path = report_dir / 'model_profile_summary.json'\nlayer_profile.to_csv(layer_path, index=False)\nmemory_trace.to_csv(liveness_path, index=False)\nsummary_path.write_text(json.dumps(profile_summary, indent=2))\n{\n    'summary': str(summary_path),\n    'layer_profile': str(layer_path),\n    'memory_trace': str(liveness_path),\n    'figures': [str(overview_path), str(opportunity_path), str(sparsity_path), str(memory_path)],\n}"
            ),
        ],
    ),
    "08_model_export.ipynb": notebook(
        "08 · Full-INT8 model export",
        "quantize with representative real images, inspect operators, verify parity, enforce size, and generate a C header",
        [
            SETUP,
            code(
                "import pandas as pd\nimport tensorflow as tf\nfrom vww_esp32.preprocessing import representative_dataset, split_manifest\n\nmanifest = pd.read_csv(paths['processed'] / 'manifest.csv')\nsplits = split_manifest(manifest)\nmodel = tf.keras.models.load_model(paths['artifacts'] / 'checkpoints' / 'best.keras')\nrepresentative = representative_dataset(splits['train'], tuple(config['preprocessing']['image_size']), config['export']['representative_samples'], config['project']['seed'])"
            ),
            code(
                "from vww_esp32.exporting import convert_full_integer\n\ntflite_path = paths['artifacts'] / 'models' / config['export']['model_filename']\nconvert_full_integer(model, representative, tflite_path)\ntflite_path"
            ),
            code(
                "import json\nfrom vww_esp32.exporting import inspect_tflite\n\nexport_info = inspect_tflite(tflite_path)\nassert export_info['input']['dtype'] == 'int8'\nassert export_info['output']['dtype'] == 'int8'\nassert export_info['size_bytes'] <= config['export']['max_model_bytes'], export_info\n(paths['artifacts'] / 'reports' / 'export_info.json').write_text(json.dumps(export_info, indent=2))\nexport_info"
            ),
            markdown(
                "Only builtin integer operators are allowed by the converter. The exact list below must match the firmware resolver; update both together."
            ),
            code("export_info['operators']"),
            code(
                "import numpy as np\nfrom vww_esp32.exporting import run_tflite\n\nsample = splits['test'].sample(n=min(100, len(splits['test'])), random_state=config['project']['seed'])\ndef host_preprocess(path):\n    decoded = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)\n    return tf.image.resize(decoded, tuple(config['preprocessing']['image_size']), method='bilinear', antialias=False).numpy()\nimages = np.stack([host_preprocess(path) for path in sample.image_path]).astype(np.float32)\nfloat_prob = model.predict(images, verbose=0).reshape(-1)\nint8_prob = run_tflite(tflite_path, images)\nparity = {'samples': len(images), 'mean_absolute_probability_error': float(np.mean(np.abs(float_prob-int8_prob))), 'max_absolute_probability_error': float(np.max(np.abs(float_prob-int8_prob))), 'decision_agreement_at_0_5': float(np.mean((float_prob >= .5) == (int8_prob >= .5)))}\nassert parity['mean_absolute_probability_error'] < 0.03, parity\nparity"
            ),
            code(
                "from vww_esp32.exporting import write_c_header\n\nheader = ROOT / 'firmware' / 'esp32_cam_vww' / 'include' / config['export']['header_filename']\nwrite_c_header(tflite_path, header, config['export']['c_array_name'])\n{'header': str(header), 'bytes': header.stat().st_size}"
            ),
        ],
    ),
    "09_report_and_deployment.ipynb": notebook(
        "09 · Report and deployment readiness",
        "assemble the model card, state evidence and limitations, and make device validation gates explicit",
        [
            SETUP,
            code(
                "import json\nimport pandas as pd\n\nmanifest = pd.read_csv(paths['processed'] / 'manifest.csv')\ndataset_summary = {'counts': manifest.groupby('split').size().to_dict(), 'class_counts': {f'{split}_{label}': int(count) for (split, label), count in manifest.groupby(['split', 'label']).size().items()}}\nmetrics = json.loads((paths['artifacts'] / 'reports' / 'test_metrics.json').read_text())\nexport_info = json.loads((paths['artifacts'] / 'reports' / 'export_info.json').read_text())\n{'dataset': dataset_summary, 'metrics': metrics, 'export': export_info}"
            ),
            code(
                "from vww_esp32.reporting import build_model_card, write_model_card\n\ncard = build_model_card(config, dataset_summary, metrics, export_info)\nmodel_card_path = write_model_card(card, ROOT / 'MODEL_CARD.md')\nprint(card)"
            ),
            markdown(
                "## ESP32-CAM release gates\n\nMark these only with measurements from the exact target board and camera.\n\n- [ ] Firmware operator resolver equals notebook 08's operator list.\n- [ ] Tensor arena high-water measured with ≥20% free margin.\n- [ ] Model flash size and total firmware partition fit.\n- [ ] Median and p95 inference latency measured over ≥1,000 frames.\n- [ ] RGB565 channel order and quantization checked with a known color target.\n- [ ] Device preprocessing compared pixel-for-pixel against a host reference.\n- [ ] At least 200 device-captured frames cover target lighting, distance, pose, and empty scenes.\n- [ ] Threshold revalidated on target-domain data without touching the COCO test result.\n- [ ] Failure behavior and downstream debounce/cooldown are documented."
            ),
            code(
                "release_checks = {\n    'full_int8_io': export_info['input']['dtype'] == export_info['output']['dtype'] == 'int8',\n    'model_size_budget': export_info['size_bytes'] <= config['export']['max_model_bytes'],\n    'held_out_metrics_present': metrics.get('samples', 0) > 0,\n    'header_generated': (ROOT / 'firmware/esp32_cam_vww/include' / config['export']['header_filename']).exists(),\n    'device_latency_measured': False,\n    'tensor_arena_measured': False,\n    'device_dataset_evaluated': False,\n}\nrelease_checks\n"
            ),
            markdown(
                "A desktop export is **not yet a production deployment**. The last three checks intentionally remain false until firmware measurements and device-captured evaluation are supplied."
            ),
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Optional notebook filenames to generate; default regenerates the full suite.",
    )
    requested = parser.parse_args().notebooks
    selected = NOTEBOOK_DEFINITIONS
    if requested:
        unknown = sorted(set(requested) - NOTEBOOK_DEFINITIONS.keys())
        if unknown:
            parser.error(f"unknown notebook(s): {', '.join(unknown)}")
        selected = {name: NOTEBOOK_DEFINITIONS[name] for name in requested}
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, payload in selected.items():
        (NOTEBOOKS / name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
