"""Reusable Keras model factory for the Day 5 churn experiment."""

from __future__ import annotations


def build_churn_ann(input_dim, hidden_units=(64, 32), dropout_rate=0.2):
    """Build a compiled binary classifier for preprocessed customer features.

    TensorFlow is imported lazily so classical-model users do not need to load
    the deep-learning runtime just to use the existing prediction service.
    """
    import tensorflow as tf

    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in hidden_units:
        layers.extend([
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
        ])
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers, name="retailmind_churn_ann")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model
