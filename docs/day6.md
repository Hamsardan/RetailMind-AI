# Day 6 — CNN Image Classification

## Objective

Add RetailMind AI's first computer-vision module: a CNN that auto-categorises catalog product images.

## Implementation

- Fashion-MNIST is the fast architecture check. Images are normalised to `[0, 1]` and reshaped to `28 × 28 × 1`.
- The CNN uses two convolution/max-pooling blocks, dropout, a dense hidden layer, and a softmax category output.
- The real-product pipeline imports `paramaggarwal/fashion-product-images-small` with `tf.keras.utils.get_file`. Keras caches and extracts it outside the repository; the pipeline then reads `styles.csv` plus `images/<id>.jpg`, predicts `subCategory`, filters classes with too few valid images, creates a deterministic stratified 80:20 split, and applies horizontal flip, rotation, and zoom only to training images.
- Evaluation prints test accuracy and a TensorFlow confusion matrix. The matrix's rows are actual categories and columns are predictions.
- The trained catalog model, ordered category mapping, and measured validation metrics are saved locally as `models/catalog_cnn_day6.keras`, `models/catalog_cnn_day6_labels.json`, and `models/catalog_cnn_day6_metrics.json`. `predict_catalog_image` returns a category and confidence for one image.

## Recorded result

No Day 6 run is recorded in this workspace. The prior Python environment is invalid because its interpreter target is missing, so no accuracy is claimed. Running the notebook prints the actual Fashion-MNIST test accuracy and confusion matrix, then optionally trains the real-image pipeline when the local image directory is present.
