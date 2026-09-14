#!/usr/bin/env python3
"""Headless entry points for the same stages demonstrated in the notebooks."""

from __future__ import annotations

import argparse

import pandas as pd

from vww_esp32.config import load_config, resolve_paths, seed_everything
from vww_esp32.data import (
    build_manifest,
    download_file,
    download_manifest_images,
    extract_zip_safely,
    validate_images,
)


def acquire(config, paths):
    archive = paths["raw"] / "annotations_trainval2017.zip"
    download_file(config["data"]["annotation_url"], archive)
    annotation_dir = paths["raw"] / "annotations"
    if not (annotation_dir / "instances_train2017.json").exists():
        extract_zip_safely(archive, paths["raw"])
    return annotation_dir


def extract(config, paths):
    annotation_dir = acquire(config, paths)
    manifest = build_manifest(
        annotation_dir / "instances_train2017.json",
        annotation_dir / "instances_val2017.json",
        paths["raw"] / "images",
        min_area_fraction=config["data"]["min_person_area_fraction"],
        validation_fraction=config["data"]["validation_fraction"],
        max_samples=config["data"]["max_samples"],
        balance_training_classes=config["data"]["balance_training_classes"],
        seed=config["project"]["seed"],
    )
    manifest = download_manifest_images(
        manifest,
        workers=config["data"]["download_workers"],
        timeout=config["data"]["download_timeout_seconds"],
    )
    output = paths["interim"] / "manifest.csv"
    manifest.to_csv(output, index=False)
    return output


def validate(config, paths):
    source = paths["interim"] / "manifest.csv"
    if not source.exists():
        source = extract(config, paths)
    clean = validate_images(pd.read_csv(source))
    clean = clean[clean["valid"]].drop(columns=["validation_error"])
    output = paths["processed"] / "manifest.csv"
    clean.to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("acquire", "extract", "validate"))
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config, root = load_config(args.config)
    paths = resolve_paths(config, root)
    seed_everything(config["project"]["seed"])
    result = {"acquire": acquire, "extract": extract, "validate": validate}[args.stage](
        config, paths
    )
    print(result)


if __name__ == "__main__":
    main()
