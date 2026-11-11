
import tensorflow as tf

def build_resnet50v2_classifier(input_shape, class_count):
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(
        scale=2.0,
        offset=-1.0,
    )(inputs)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        class_count,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="retailmind_resnet50v2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model
