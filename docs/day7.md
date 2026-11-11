# Day 7 — Transfer Learning for Product Image Classification

## Objective

Improve the Day 6 product-image classifier by replacing its from-scratch CNN backbone with a pretrained ResNet50V2 model. This increment covers the first two Day 7 tasks: training a new classification head with the pretrained backbone frozen, then fine-tuning the top backbone layers and comparing the results with Day 6.

YOLO and object detection are outside the scope of this increment and are not implemented.

## Dataset

The experiment reuses the Day 6 Fashion Product Images dataset and the same deterministic train/validation split. The target is the `subCategory` column.

- Rows retained after image/label validation and minimum-class filtering: 43,996
- Categories retained: 27
- Training images: 35,186
- Validation images: 8,810
- Image size: `128 × 128`
- Batch size: 16
- Split seed: 42

Using the same dataset split makes the Day 6 and Day 7 validation accuracies directly comparable.

## Implementation

### Pretrained classifier

- Backbone: `tf.keras.applications.ResNet50V2`
- Pretrained weights: ImageNet
- Original classification head: removed with `include_top=False`
- Input preprocessing: the dataset supplies images in `[0, 1]`; a Keras `Rescaling(scale=2.0, offset=-1.0)` layer converts them to `[-1, 1]` for ResNet50V2.
- New classification head: Global Average Pooling, Dropout (`0.3`), and a Dense softmax layer with 27 outputs.
- Loss: sparse categorical cross-entropy
- Optimizer: Adam

### Stage 1 — Frozen backbone

The ResNet50V2 backbone was frozen and only the new classification head was trained.

- Epochs: 5
- Learning rate: `1e-3`
- Early stopping: monitor `val_loss`, patience 3, restore best weights

### Stage 2 — Fine-tuning

The final 30 ResNet50V2 layers were marked trainable. Batch Normalization layers were kept frozen, leaving 22 trainable layers in the ResNet50V2 backbone. The classification head remained trainable.

- Epochs: 5
- Learning rate: `1e-5`
- Early stopping: monitor `val_loss`, patience 3, restore best weights

The lower learning rate limits changes to the pretrained features during fine-tuning.

## Results

All three results below use the same 8,810-image validation split.

| Model                              | Validation accuracy |
| ---------------------------------- | ------------------: |
| Day 6 CNN (trained from scratch)   |              84.93% |
| Day 7 ResNet50V2 (frozen backbone) |              91.37% |
| Day 7 ResNet50V2 (fine-tuned)      |          **96.04%** |

Fine-tuning improved validation accuracy by **11.11 percentage points** over the Day 6 CNN and by **4.67 percentage points** over the frozen-backbone model.

The frozen-backbone `fit()` log showed 92.04% validation accuracy in its fifth epoch. However, EarlyStopping restored the weights from the epoch with the lowest validation loss (epoch 2); evaluating that restored model produced the recorded 91.37% accuracy. The table reports the post-training `evaluate()` results, not the last epoch's transient metric.

## Environment and runtime

- TensorFlow: `2.21.0`
- Environment: native Windows
- TensorFlow reported that GPU support was unavailable in this environment, so training ran without TensorFlow GPU acceleration.
- The reported per-epoch durations sum to approximately 4 hours 14 minutes across both five-epoch stages, excluding setup and evaluation.

## Saved artifacts

The Day 7 notebook saves the fine-tuned model and its category mapping and metrics separately from the Day 6 baseline:

- `models/catalog_resnet50v2_day7.keras`
- `models/catalog_resnet50v2_day7_labels.json`
- `models/catalog_resnet50v2_day7_metrics.json`

The Day 6 model and its artifacts remain unchanged and are retained as the comparison baseline.

## Conclusion

On the same product-image validation split, ResNet50V2 transfer learning outperformed the Day 6 CNN. Fine-tuning the top backbone layers raised validation accuracy to 96.04%, an improvement of 11.11 percentage points over the from-scratch CNN. The fine-tuned ResNet50V2 is the stronger catalog classifier based on this experiment's validation accuracy.
