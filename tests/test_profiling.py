import numpy as np
import tensorflow as tf

from vww_esp32.profiling import activation_liveness, estimate_layer_macs, profile_model


def test_conv_and_dense_macs_are_counted():
    inputs = tf.keras.Input((8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, use_bias=False)(x)
    model = tf.keras.Model(inputs, outputs)

    assert estimate_layer_macs(model.layers[1]) == 8 * 8 * 4 * 3 * 3 * 3
    assert estimate_layer_macs(model.layers[-1]) == 4 * 2


def test_profile_totals_and_liveness_are_consistent():
    inputs = tf.keras.Input((4, 4, 2))
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(inputs)
    outputs = tf.keras.layers.Conv2D(3, 1, use_bias=False)(x)
    model = tf.keras.Model(inputs, outputs)

    layers, liveness, summary = profile_model(model, training_batch_size=2)

    assert summary["parameters"] == model.count_params()
    assert summary["estimated_macs_batch1"] == 4 * 4 * 2 * 3 * 3 + 4 * 4 * 3 * 2
    assert summary["peak_live_activation_float32_bytes_batch1"] > 0
    assert layers.parameters.sum() == model.count_params()
    assert not liveness.empty


def test_activation_liveness_handles_branching_graph():
    inputs = tf.keras.Input((4,))
    left = tf.keras.layers.Dense(4)(inputs)
    right = tf.keras.layers.Dense(4)(inputs)
    outputs = tf.keras.layers.Add()([left, right])
    model = tf.keras.Model(inputs, outputs)

    trace = activation_liveness(model)

    assert trace.live_tensors.max() >= 3
    assert trace.live_float32_bytes.max() >= 3 * 4 * 4


def test_profile_reports_exact_and_near_zero_kernel_fractions():
    inputs = tf.keras.Input((3,))
    outputs = tf.keras.layers.Dense(2, use_bias=False)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.layers[-1].set_weights(
        [np.array([[0.0, 5e-7], [1e-4, 1.0], [-1.0, 2.0]], dtype=np.float32)]
    )

    _, _, summary = profile_model(model, near_zero_threshold=1e-6)

    assert summary["global_kernel_sparsity"] == 1 / 6
    assert summary["global_kernel_near_zero_fraction"] == 2 / 6
