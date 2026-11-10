"""Day 6 TensorFlow/Keras catalog-image classification module.

The real-data loader reads the original Kaggle download directly:
``fashion-product-images-small/styles.csv`` and its ``images/<id>.jpg`` files.
No manual folder rearrangement or non-TensorFlow data library is required.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


FASHION_MNIST_LABELS = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
FASHION_PRODUCT_IMAGES_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "paramaggarwal/fashion-product-images-small"
)


def download_fashion_product_images():
    """Fetch the public Kaggle dataset through Keras and return its root.

    Keras caches and extracts the archive outside this repository, so catalog
    source images are not copied into Git or required as a manual download.
    """
    import tensorflow as tf

    archive = Path(tf.keras.utils.get_file(
        fname="fashion-product-images-small.zip",
        origin=FASHION_PRODUCT_IMAGES_URL,
        extract=True,
        cache_subdir="retailmind_day6",
    ))
    for metadata_path in archive.parent.rglob("styles.csv"):
        if (metadata_path.parent / "images").is_dir():
            return metadata_path.parent
    raise FileNotFoundError(
        "Keras downloaded the archive but no styles.csv plus images/ directory "
        "was found after extraction."
    )


def build_catalog_cnn(input_shape, class_count):
    """Create the Day 6 conv → pool → dropout → dense classifier."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ], name="retailmind_catalog_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fashion_mnist_data():
    """Load Fashion-MNIST as normalized 28×28 grayscale tensors."""
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = tf.expand_dims(tf.cast(x_train, tf.float32) / 255.0, -1)
    x_test = tf.expand_dims(tf.cast(x_test, tf.float32) / 255.0, -1)
    return (x_train, y_train), (x_test, y_test)


def _product_records(dataset_dir, label_column, min_examples_per_class):
    """Read valid image paths and labels from the raw Kaggle metadata CSV."""
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "styles.csv"
    image_dir = dataset_dir / "images"
    if not metadata_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(
            "Expected the Kaggle layout: "
            f"{metadata_path} and {image_dir}/<id>.jpg"
        )

    rows = []
    with metadata_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            image_id = (row.get("id") or "").strip()
            label = (row.get(label_column) or "").strip()
            image_path = image_dir / f"{image_id}.jpg"
            if image_id and label and image_path.is_file():
                rows.append((str(image_path), label))
    if not rows:
        raise ValueError("No metadata rows had a matching product image.")

    counts = Counter(label for _, label in rows)
    retained = {label for label, count in counts.items() if count >= min_examples_per_class}
    rows = [(path, label) for path, label in rows if label in retained]
    if len(retained) < 2:
        raise ValueError("At least two label categories need matching image files.")
    return rows, sorted(retained), counts


def fashion_product_datasets(
    dataset_dir,
    label_column="subCategory",
    image_size=(128, 128),
    batch_size=32,
    min_examples_per_class=100,
    seed=42,
):
    """Build augmented, stratified tf.data train/validation datasets.

    Labels default to the Kaggle dataset's ``subCategory`` field. Classes with
    fewer than ``min_examples_per_class`` matching files are excluded so each
    retained category has enough examples for training and validation.
    """
    import tensorflow as tf

    rows, class_names, counts = _product_records(
        dataset_dir, label_column, min_examples_per_class
    )
    class_index = {name: index for index, name in enumerate(class_names)}
    grouped = defaultdict(list)
    for path, label in rows:
        grouped[label].append(path)

    train_paths, train_labels, validation_paths, validation_labels = [], [], [], []
    # Sorting plus every fifth image produces a reproducible stratified 80:20 split.
    for label in class_names:
        for index, path in enumerate(sorted(grouped[label])):
            target_paths = validation_paths if index % 5 == 0 else train_paths
            target_labels = validation_labels if index % 5 == 0 else train_labels
            target_paths.append(path)
            target_labels.append(class_index[label])

    def decode(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32) / 255.0, label

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="catalog_augmentation")
    autotune = tf.data.AUTOTUNE
    train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train = train.shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
    train = train.map(decode, num_parallel_calls=autotune).batch(batch_size)
    train = train.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    validation = tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels))
    validation = validation.map(decode, num_parallel_calls=autotune)
    validation = validation.batch(batch_size).prefetch(autotune)
    summary = {
        "label_column": label_column,
        "source_rows": len(rows),
        "train_rows": len(train_paths),
        "validation_rows": len(validation_paths),
        "class_counts": {name: counts[name] for name in class_names},
    }
    return train, validation, class_names, summary


def save_catalog_classifier(model, class_names, model_path):
    """Persist a Keras catalog model and its ordered category mapping."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path = model_path.with_name(f"{model_path.stem}_labels.json")
    model.save(model_path)
    labels_path.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")
    return model_path, labels_path


def save_catalog_metrics(model_path, validation_accuracy, dataset_summary):
    """Save the measured validation accuracy beside the catalog model."""
    model_path = Path(model_path)
    metrics_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    metrics = {"validation_accuracy": float(validation_accuracy), **dataset_summary}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def predict_catalog_image(image_path, model_path, image_size=(128, 128)):
    """Classify one product image with its persisted model and label mapping."""
    import tensorflow as tf

    model_path = Path(model_path)
    labels_path = model_path.with_name(f"{model_path.stem}_labels.json")
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    image = tf.keras.utils.img_to_array(image)[None, ...] / 255.0
    probabilities = tf.keras.models.load_model(model_path).predict(image, verbose=0)[0]
    index = int(tf.argmax(probabilities))
    return {"category": class_names[index], "confidence": float(probabilities[index])}
