from __future__ import annotations

from pathlib import Path

import pandas as pd


def split_manifest(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: manifest[manifest["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }


def make_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int] = (96, 96),
    batch_size: int = 64,
    training: bool = False,
    seed: int = 42,
    cache: bool | str | Path = False,
):
    """Create a deterministic tf.data pipeline; pixel normalization stays in the model."""
    import tensorflow as tf

    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["label"].astype("float32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(min(len(frame), 4096), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        # Half-pixel bilinear geometry is mirrored by the firmware reference kernel.
        image = tf.image.resize(image, image_size, method="bilinear", antialias=False)
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.float32)
        image.set_shape((*image_size, 3))
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache(str(cache) if isinstance(cache, (str, Path)) else "")
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def representative_dataset(frame: pd.DataFrame, image_size=(96, 96), samples=500, seed=42):
    """Yield unbatched float32 input samples for post-training INT8 calibration."""
    subset = frame.sample(n=min(samples, len(frame)), random_state=seed)
    dataset = make_dataset(subset, image_size=image_size, batch_size=1, training=False)
    for image, _ in dataset:
        yield [image]
