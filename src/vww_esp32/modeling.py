from __future__ import annotations

from pathlib import Path


def build_tiny_mobilenet_v1(
    input_shape=(96, 96, 3),
    alpha: float = 0.25,
    dropout: float = 0.1,
    l2: float = 1e-5,
    camera_augmentation: bool = False,
    camera_augmentation_strength: float = 1.0,
    augmentation_seed: int | None = None,
):
    """Compressed MobileNetV1 binary classifier using TFLite Micro friendly operators.

    The convolutional trunk is a branch-free sequence of depthwise 3x3 and
    pointwise 1x1 convolutions, following the Visual Wake Words paper.  It
    intentionally removes three late blocks from canonical MobileNetV1 because
    those blocks add latency on the original ESP32 without reducing the large
    early activation maps.  Optional caamera augmentation is active only during
    training and therefore adds no operators to the exported inference graph.
    """
    import tensorflow as tf

    regularizer = tf.keras.regularizers.l2(l2)

    def channels(value: int) -> int:
        return max(8, int(value * alpha))

    def conv_bn_relu(x, filters: int, stride: int, name: str):
        x = tf.keras.layers.Conv2D(
            channels(filters),
            3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
            name=f"{name}_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)

    def depthwise_separable(x, filters: int, stride: int, name: str):
        x = tf.keras.layers.DepthwiseConv2D(
            3,
            strides=stride,
            padding="same",
            use_bias=False,
            depthwise_regularizer=regularizer,
            name=f"{name}_dw",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_dw_bn")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_dw_relu6")(x)
        x = tf.keras.layers.Conv2D(
            channels(filters),
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
            name=f"{name}_pw",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_pw_bn")(x)
        return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_pw_relu6")(x)

    inputs = tf.keras.Input(shape=input_shape, name="image", dtype=tf.float32)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(inputs)
    x = tf.keras.layers.RandomFlip(
        "horizontal", seed=augmentation_seed, name="augment_flip"
    )(x)
    x = tf.keras.layers.RandomTranslation(
        0.05,
        0.05,
        fill_mode="reflect",
        seed=None if augmentation_seed is None else augmentation_seed + 1,
        name="augment_shift",
    )(x)
    if camera_augmentation:
        strength = max(0.0, float(camera_augmentation_strength))
        # The OV3660 stream is often desaturated, low-contrast, noisy, and
        # unevenly exposed.  Train across those conditions while keeping the
        # validation/test images untouched.
        x = tf.keras.layers.RandomBrightness(
            0.25 * strength,
            value_range=(-1.0, 1.0),
            seed=None if augmentation_seed is None else augmentation_seed + 2,
            name="augment_brightness",
        )(x)
        x = tf.keras.layers.RandomContrast(
            0.35 * strength,
            value_range=(-1.0, 1.0),
            seed=None if augmentation_seed is None else augmentation_seed + 3,
            name="augment_contrast",
        )(x)
        x = tf.keras.layers.RandomSaturation(
            min(1.0, 0.9 * strength),
            value_range=(-1.0, 1.0),
            seed=None if augmentation_seed is None else augmentation_seed + 4,
            name="augment_saturation",
        )(x)
        x = tf.keras.layers.GaussianNoise(
            0.04 * strength,
            seed=None if augmentation_seed is None else augmentation_seed + 5,
            name="augment_sensor_noise",
        )(x)
    x = conv_bn_relu(x, 32, 2, "stem")
    for index, (filters, stride) in enumerate(
        [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)]
    ):
        x = depthwise_separable(x, filters, stride, f"block{index + 1}")
    for index in range(3):
        x = depthwise_separable(x, 512, 1, f"block{index + 7}")
    x = depthwise_separable(x, 1024, 2, "block10")
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="person_probability")(x)
    return tf.keras.Model(inputs, outputs, name=f"tiny_mobilenet_v1_{alpha:g}")


def compile_model(model, learning_rate=1e-3, label_smoothing=0.05):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def training_callbacks(artifacts: str | Path, monitor="val_pr_auc", patience=7, lr_patience=3):
    import tensorflow as tf

    artifacts = Path(artifacts)
    checkpoints = artifacts / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoints / "best.keras", monitor=monitor, mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.3, patience=lr_patience, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(artifacts / "logs" / "training_history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
