import json

from vww_esp32.data import annotation_frame, build_manifest, coco_download_url


def tiny_coco(image_ids):
    return {
        "images": [
            {
                "id": image_id,
                "file_name": f"{image_id}.jpg",
                "width": 100,
                "height": 100,
                "coco_url": f"https://example.invalid/{image_id}.jpg",
                "license": 1,
            }
            for image_id in image_ids
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "cat"}],
        "annotations": [
            # Segmentation area intentionally differs; VWW must use bbox area.
            {"image_id": image_ids[0], "category_id": 1, "area": 1, "bbox": [0, 0, 20, 30]},
            {"image_id": image_ids[1], "category_id": 1, "area": 400, "bbox": [0, 0, 20, 20]},
            {"image_id": image_ids[1], "category_id": 2, "area": 5000, "bbox": [0, 0, 50, 100]},
        ],
    }


def test_annotation_frame_computes_person_fraction():
    frame = annotation_frame(tiny_coco([1, 2]))
    assert frame.set_index("image_id").loc[1, "max_person_area_fraction"] == 0.06
    assert frame.set_index("image_id").loc[2, "person_count"] == 1


def test_coco_url_uses_tls_valid_s3_path_style_endpoint():
    legacy = "http://images.cocodataset.org/train2017/000000000009.jpg"
    assert coco_download_url(legacy) == (
        "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000000009.jpg"
    )


def test_manifest_threshold_and_splits_are_deterministic(tmp_path):
    train_path, test_path = tmp_path / "train.json", tmp_path / "test.json"
    train_path.write_text(json.dumps(tiny_coco(list(range(1, 31)))))
    test_path.write_text(json.dumps(tiny_coco([101, 102])))
    kwargs = {
        "train_annotations": train_path,
        "test_annotations": test_path,
        "image_root": tmp_path / "images",
        "min_area_fraction": 0.05,
        "validation_fraction": 0.2,
        "max_samples": {"train": None, "val": None, "test": None},
        "seed": 7,
    }
    first, second = build_manifest(**kwargs), build_manifest(**kwargs)
    assert first[["image_id", "split"]].equals(second[["image_id", "split"]])
    assert first.loc[first.image_id == 1, "label"].item() == 1
    assert first.loc[first.image_id == 2, "label"].item() == 0
    assert set(first[first.source_split == "val2017"].split) == {"test"}
