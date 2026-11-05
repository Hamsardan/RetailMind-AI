# RetailMind AI

RetailMind AI is an incremental retail intelligence platform. Its current module scores customer churn risk using the Telco Customer Churn dataset (`blastchar/telco-customer-churn`).

## Churn model

All reported holdout metrics use the same stratified 1,409-customer test set. Churn recall is the primary model-selection metric.

| Milestone | Selected model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Day 1 baseline | Logistic Regression | 80.55% | 65.72% | 55.88% | 84.21% |
| Day 3 tuned | Logistic Regression (`C=1`, balanced weights) | 73.81% | 50.43% | 78.34% | 84.16% |
| Day 4 production | Day 3 champion, retained after ensemble comparison | 73.81% | 50.43% | 78.34% | 84.16% |

Day 4 used five-fold stratified cross-validation on the training split. The Day 3 champion led with **0.8013 mean recall**; XGBoost reached 0.7659 and Random Forest 0.7157. Feature selection did not improve either ensemble. The selected production artifact is `models/churn_v4.joblib`.

The complete comparison, feature engineering, and final evaluation live in [`notebooks/churn_analysis.ipynb`](notebooks/churn_analysis.ipynb). The saved pipeline freezes `TotalCharges` cleaning, scaling, and categorical encoding; the Day 4 ensemble candidates also include tenure, charge, service, and contract features.

## Run

Install the project dependencies with `pip install -r requirements.txt`, then open and run the notebook. It trains and saves `models/churn_v4.joblib`.

For single-customer or batch predictions, call `predict` from [`src/predict.py`](src/predict.py) with a dictionary or DataFrame containing the model's input columns. It returns a churn label and churn probability. The service uses the Day 3 artifact until the Day 4 artifact exists.
