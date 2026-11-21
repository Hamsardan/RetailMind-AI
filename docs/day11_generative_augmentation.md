# RetailMind AI — Day 11

## VAE Image Augmentation

### Objective
Generate synthetic images for a minority product category
and measure their effect on image classification accuracy.

### Dataset
Fashion Product Images (Small), Kaggle

Dataset location: Local Keras cache
Classification labels: subCategory
Number of categories: 27

Real training images: 6812
Validation images: 1478
Test images: 1478

Training samples per class capped at: 300
Evaluation samples per class capped at: 80

### Minority Category Selection
Category: Free Gifts
Original category images: 104
Available training images: 84
Reference confusion matrix: Day 11 baseline validation

### Generative Model
Model: Convolutional Variational Autoencoder
Framework: TensorFlow / Keras
VAE image resolution: 64 x 64
Latent dimensions: 32
Epochs: 40

Generated candidates: 64
Manually approved samples: 10

### Classifier
Architecture: Original Day 6 Catalog CNN
Classifier resolution: 128 x 128
Training epochs (maximum): 5

### Overall Results
Baseline accuracy: 0.3417
Augmented accuracy: 0.3701
Accuracy change: +0.0284

Baseline macro recall: 0.2711
Augmented macro recall: 0.3024

### Target Category Results
Category: Free Gifts
Before augmentation recall: 0.0000
After augmentation recall: 0.0000
Recall change: +0.0000

### Weekly Assessment 04
The full per-class comparison is saved in:
docs/day11_per_class_results.csv

### Limitations
- This is a controlled Day 11 CNN experiment, not a
  direct comparison of previously saved Day 6/7 scores.
- The training sample limit reduces resource requirements.
- Synthetic samples are selected through manual inspection.
- The generator uses 64x64 images, which are resized to
  128x128 for CNN training.
- Single-run changes may reflect training randomness.
- Per-class recall is not equivalent to demographic fairness.
- If the earlier confusion matrix was unavailable,
  selection used Day 11 validation predictions instead.

### Conclusion
Observed overall accuracy change: +0.0284
Observed target category recall change:
+0.0000

These measurements describe this experiment only.
