from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm


def coco_download_url(url: str) -> str:
    """Map COCO's legacy hostname to its TLS-valid public S3 bucket URL."""
    parsed = urlsplit(str(url))
    if parsed.hostname == "images.cocodataset.org":
        rewritten = f"https://s3.amazonaws.com/images.cocodataset.org{parsed.path}"
        return f"{rewritten}?{parsed.query}" if parsed.query else rewritten
    return str(url)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: str | Path, timeout: int = 60) -> Path:
    """Stream a URL to disk, resuming a partial `.part` file when supported."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {"Range": f"bytes={partial.stat().st_size}-"} if partial.exists() else {}
    mode = "ab" if headers else "wb"
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        if response.status_code == 200 and headers:
            mode = "wb"
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            partial.open(mode) as handle,
            tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress,
        ):
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)
    return destination


def extract_zip_safely(archive: str | Path, destination: str | Path) -> list[Path]:
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            bundle.extract(member, destination)
            extracted.append(target)
    return extracted


def load_coco(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def annotation_frame(coco: dict, target_category: str = "person") -> pd.DataFrame:
    """Convert COCO detection JSON to one image-level row with VWW diagnostics."""
    category_ids = {c["id"] for c in coco["categories"] if c["name"] == target_category}
    if not category_ids:
        raise ValueError(f"Category {target_category!r} does not exist in annotations")
    image_by_id = {image["id"]: image for image in coco["images"]}
    person_stats: dict[int, dict[str, float]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in category_ids or ann.get("iscrowd", 0):
            continue
        image = image_by_id[ann["image_id"]]
        # VWW is defined using the bounding-box area, not COCO's segmented-mask area.
        fraction = float(ann["bbox"][2] * ann["bbox"][3]) / (image["width"] * image["height"])
        stats = person_stats.setdefault(ann["image_id"], {"count": 0, "max_fraction": 0.0})
        stats["count"] += 1
        stats["max_fraction"] = max(stats["max_fraction"], fraction)
    rows = []
    for image_id, image in image_by_id.items():
        stats = person_stats.get(image_id, {"count": 0, "max_fraction": 0.0})
        rows.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "aspect_ratio": image["width"] / image["height"],
                "license_id": image.get("license"),
                "coco_url": image.get("coco_url") or image.get("flickr_url"),
                "person_count": int(stats["count"]),
                "max_person_area_fraction": float(stats["max_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def _stable_fraction(value: int, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / (2**64 - 1)


def _sample_frame(
    frame: pd.DataFrame, limit: int | None, seed: int, balance: bool = False
) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.copy()
    if not balance:
        return frame.sample(n=limit, random_state=seed)
    per_class = limit // 2
    chunks = []
    for label, group in frame.groupby("label"):
        chunks.append(group.sample(n=min(per_class, len(group)), random_state=seed + int(label)))
    result = pd.concat(chunks)
    remaining = limit - len(result)
    if remaining > 0:
        unused = frame.drop(result.index)
        result = pd.concat(
            [result, unused.sample(n=min(remaining, len(unused)), random_state=seed)]
        )
    return result


def build_manifest(
    train_annotations: str | Path,
    test_annotations: str | Path,
    image_root: str | Path,
    min_area_fraction: float = 0.005,
    validation_fraction: float = 0.15,
    max_samples: dict[str, int | None] | None = None,
    balance_training_classes: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Build deterministic train/val/test image-level VWW labels from COCO."""
    train = annotation_frame(load_coco(train_annotations))
    test = annotation_frame(load_coco(test_annotations))
    train["split"] = train["image_id"].map(
        lambda value: "val" if _stable_fraction(value, seed) < validation_fraction else "train"
    )
    train["source_split"] = "train2017"
    test["split"], test["source_split"] = "test", "val2017"
    manifest = pd.concat([train, test], ignore_index=True)
    manifest["label"] = (manifest["max_person_area_fraction"] >= min_area_fraction).astype("int8")
    limits = max_samples or {}
    selected = []
    for offset, split in enumerate(("train", "val", "test")):
        subset = manifest[manifest["split"] == split]
        selected.append(
            _sample_frame(
                subset,
                limits.get(split),
                seed + offset,
                balance=balance_training_classes and split == "train",
            )
        )
    manifest = pd.concat(selected, ignore_index=True)
    image_root = Path(image_root)
    manifest["image_path"] = manifest.apply(
        lambda row: str(image_root / row["source_split"] / row["file_name"]), axis=1
    )
    return manifest.sort_values(["split", "image_id"]).reset_index(drop=True)


def _download_image(row: dict, timeout: int, retries: int = 3) -> tuple[int, str | None]:
    target = Path(row["image_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with Image.open(target) as image:
                image.verify()
            return row["image_id"], None
        except OSError:
            target.unlink(missing_ok=True)
    error = None
    for attempt in range(retries):
        try:
            image_url = coco_download_url(row["coco_url"])
            with requests.get(image_url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                partial = target.with_suffix(target.suffix + ".part")
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response.raw, handle)
                with Image.open(partial) as image:
                    image.verify()
                partial.replace(target)
            return row["image_id"], None
        except (requests.RequestException, OSError) as exc:
            error = str(exc)
            time.sleep(2**attempt)
    return row["image_id"], error


def download_manifest_images(
    manifest: pd.DataFrame, workers: int = 8, timeout: int = 45
) -> pd.DataFrame:
    """Download only manifest-selected COCO images and report failures."""
    records = manifest.to_dict("records")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_image, row, timeout) for row in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="COCO images"):
            image_id, error = future.result()
            if error:
                failures.append({"image_id": image_id, "error": error})
    result = manifest.copy()
    failed_ids = {item["image_id"] for item in failures}
    result["downloaded"] = ~result["image_id"].isin(failed_ids)
    result.attrs["failures"] = failures
    return result


def validate_images(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Validate"):
        record = row._asdict()
        try:
            with Image.open(record["image_path"]) as image:
                image.verify()
            record.update(valid=True, validation_error="")
        except (OSError, FileNotFoundError) as exc:
            record.update(valid=False, validation_error=str(exc))
        rows.append(record)
    return pd.DataFrame(rows)
