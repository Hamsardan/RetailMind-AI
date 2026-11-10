# Day 3 — Multi-Model Tuning & Bias-Variance

Selection used 5-fold cross-validation on the original Day 1 training partition, with churn-class recall as the scoring metric. The original stratified 20% holdout was not used during tuning and was evaluated once after selecting the highest-CV-recall model.

Search grids: Logistic Regression tested `C=[0.1, 1, 10]` and `class_weight=[None, balanced]`; Decision Tree tested `max_depth=[None, 5, 10, 20]`, `min_samples_leaf=[1, 5, 10]`, and both class-weight settings; Random Forest tested `n_estimators=[100, 250]`, `max_depth=[None, 10, 20]`, `min_samples_leaf=[1, 5]`, and both class-weight settings; SVM tested RBF `C=[0.1, 1, 10]` and both class-weight settings. All searches used 5 folds.

## Search results

| Model | Mean train recall at best setting | Mean CV recall | Train–CV gap | Selected parameters |
|---|---:|---:|---:|---|
| Logistic Regression | 0.8052 | 0.8040 | 0.0012 | `{'classifier__C': 1, 'classifier__class_weight': 'balanced'}` |
| Decision Tree | 0.8227 | 0.7846 | 0.0381 | `{'classifier__class_weight': 'balanced', 'classifier__max_depth': 5, 'classifier__min_samples_leaf': 5}` |
| Random Forest | 0.8911 | 0.7732 | 0.1179 | `{'classifier__class_weight': 'balanced', 'classifier__max_depth': 10, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 250}` |
| SVM | 0.8087 | 0.7967 | 0.0120 | `{'classifier__C': 0.1, 'classifier__class_weight': 'balanced', 'classifier__kernel': 'rbf'}` |

## Selected model: Logistic Regression

The champion was selected by the highest mean 5-fold validation recall (0.8040). Selected search parameters: `C=1`, `class_weight='balanced'`.

## Final holdout evaluation

| Accuracy | Precision | Recall | ROC-AUC |
|---:|---:|---:|---:|
| 0.7381 | 0.5043 | 0.7834 | 0.8416 |

## Learning curve (recall)

| Training rows per fold | Train recall | Validation recall |
|---:|---:|---:|
| 901 | 0.8196 | 0.7953 |
| 1802 | 0.8229 | 0.8033 |
| 2704 | 0.8216 | 0.7987 |
| 3605 | 0.8140 | 0.8027 |
| 4507 | 0.8050 | 0.8040 |

## Bias-variance interpretation

Train and validation recall converge near 0.80 with a very small gap, suggesting limited variance for this metric. Validation recall changes little across the largest training sizes, so more data alone is unlikely to produce a large gain. The curve does not show low recall that would clearly indicate high bias; the model has reasonable churn recall, with the precision tradeoff shown in the holdout results.

Saved complete preprocessing-plus-classifier pipeline to `models/churn_v3.joblib`.
The baseline model and its Day 1/2 artifacts are retained for comparison.
