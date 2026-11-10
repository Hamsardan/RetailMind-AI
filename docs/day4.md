# Day 4 — Feature Engineering and Ensembles

## Objective

Add domain features to the Telco churn data and compare feature selection, bagging, and boosting against the tuned Day 3 champion.

## Method

- Engineer tenure, charge, service-count, contract, and customer relationship features in `src/feature_engineering.py`.
- Keep cleaning, scaling, encoding, optional feature selection, and classification inside each pipeline so cross-validation fits transformations within each fold.
- Compare Random Forest and XGBoost variants with and without feature selection against the Day 3 logistic champion using five-fold stratified cross-validation on the training split.
- Rank by churn recall, with ROC-AUC as the tie-breaker; use the held-out test set only for the selected pipeline's final evaluation.

## Decision and artifact

The Day 3 champion remained selected. Feature selection and the ensemble candidates did not improve the cross-validation ranking. Its holdout result remains 73.81% accuracy, 50.43% precision, 78.34% recall, and 84.16% ROC-AUC. The complete inference pipeline is saved as `models/churn_v4.joblib`.

The increment rebuilds the Day 3 champion, so it can run independently.
