# RetailMind AI

RetailMind AI is an incremental retail intelligence project. It currently includes customer-churn prediction and a CNN product-image categorisation module.

## Repository layout

```text
data/       source data
docs/       one concise report per project day
models/     saved, reusable model artifacts
notebooks/  one independently runnable notebook per day
src/        shared churn and catalog-classification utilities
```

Each notebook prepares its own data and fixed train/test split. Run a day's notebook without running earlier notebooks first. Day 4 rebuilds the recorded Day 3 champion for its comparison; Day 5 does the same for its benchmark.

## Churn model results

All reported holdout metrics use the same stratified 1,409-customer test set. Churn recall is the primary model-selection metric.

| Milestone | Selected model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Day 1 baseline | Logistic Regression | 80.55% | 65.72% | 55.88% | 84.21% |
| Day 3 tuned | Logistic Regression (`C=1`, balanced weights) | 73.81% | 50.43% | 78.34% | 84.16% |
| Day 4 production | Day 3 champion retained after ensemble comparison | 73.81% | 50.43% | 78.34% | 84.16% |
| Day 5 challenger | Keras ANN | 75.44% | 52.68% | 73.53% | 83.83% |

Day 4 compared the Day 3 champion with Random Forest and XGBoost, with and without feature selection. Five-fold stratified cross-validation used only the training split. Neither the ensembles nor feature selection displaced the champion. The saved production artifact is `models/churn_v4.joblib`.

Day 5's ANN scored 4.81 percentage points lower on recall and 0.33 points lower on ROC-AUC than the classical champion, so it remains opt-in. It is saved as `models/churn_ann_day5.keras` with `models/churn_ann_preprocessor.joblib`.

### When to use which model

For the tabular churn workflow, use the classical champion: it has higher measured recall (78.34% versus 73.53%) and ROC-AUC (84.16% versus 83.83%) than the ANN.

The daily notebooks are independent: each prepares its own inputs and saved artifacts.

## Install and run

```bash
pip install -r requirements.txt
```

Open and run the notebook for the increment you want. Day 5 downloads MNIST on first use. Day 6 downloads Fashion-MNIST and the public Kaggle Fashion Product Images archive automatically through Keras, caching the data outside this repository.

## Catalog image classification

Day 6 validates its CNN on Fashion-MNIST, then automatically imports and trains on the raw Kaggle product dataset. It classifies `subCategory` labels, uses a deterministic per-category 80:20 split, and saves its model, category mapping, and measured validation accuracy locally.

| Model | Accuracy |
|---|---:|
| Day 6 catalog CNN (Fashion Product Images validation split) | 84.93% |

Once trained, classify one image directly:

```python
from src.catalog_classifier import predict_catalog_image

predict_catalog_image(
    "path/to/product-image.jpg",
    "models/catalog_cnn_day6.keras",
)
```

## Prediction service

The Day 1 prediction interface remains available in `src/predict.py`. It accepts one customer as a dictionary or a batch as a DataFrame and returns a churn label and churn probability. By default it loads the latest saved classical champion. Choose a saved increment with `model_name="day1"` through `"day4"`, or opt into the ANN with `model_name="ann"` after running Day 5.

```python
import pandas as pd
from src.predict import predict

sample = pd.read_csv("data/churn.csv").drop(columns=["customerID", "Churn"]).iloc[0]
result = predict(sample.to_dict())
```

Provide all model input columns. The ANN path requires both Day 5 artifacts to exist.
