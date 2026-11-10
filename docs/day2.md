# Day 2 — Decision Tree, Random Forest & SVM

## Objective

Evaluate alternative classification algorithms for customer churn prediction and identify the strongest performer against the Day 1 Logistic Regression baseline.

## Models Evaluated

Three classifiers were trained using the same preprocessing pipeline:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Support Vector Machine (SVM)

All models used the same training and testing datasets.

## Preprocessing

Reused the Day 1 preprocessing strategy:

- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- ColumnTransformer for combined preprocessing.
- Scikit-learn Pipeline for preprocessing and classification.

## Model Configuration

| Model         | Configuration                     |
| ------------- | --------------------------------- |
| Decision Tree | random_state=42                   |
| Random Forest | random_state=42, n_jobs=-1        |
| SVM           | probability=True, random_state=42 |

Default classifier hyperparameters were used.

## Evaluation Results

All models were evaluated on the same held-out test set of 1,409 samples.

| Model                       | Accuracy | Precision | Recall | ROC-AUC |
| --------------------------- | -------: | --------: | -----: | ------: |
| Logistic Regression (Day 1) |   80.55% |    65.72% | 55.88% |  84.21% |
| Decision Tree               |   72.11% |    47.55% | 49.20% |  64.77% |
| Random Forest               |   78.64% |    62.54% | 48.66% |  81.85% |
| SVM                         |   79.13% |    64.18% | 48.40% |  79.02% |

## Champion Selection

Recall was chosen as the primary selection criterion because identifying customers likely to churn is the primary business objective.

### Day 2 Challenger Champion: Decision Tree

Among the three newly evaluated models, Decision Tree achieved the highest recall of 49.20%.

However, none of the three challengers improved upon the Day 1 Logistic Regression baseline.

### Overall Platform Champion: Logistic Regression

Logistic Regression remains the production champion with:

- Accuracy: 80.55%
- Precision: 65.72%
- Recall: 55.88%
- ROC-AUC: 84.21%

The Decision Tree model is persisted separately for future experimentation.

## Model Artifacts

- `models/churn_pipeline.joblib` — Day 1 Logistic Regression baseline and current overall champion.
- `models/churn_day2_champion.joblib` — Day 2 Decision Tree challenger champion.

## Conclusion

The three alternative algorithms did not outperform the initial Logistic Regression baseline using their default configurations.

The results establish a benchmark for future experimentation and highlight the need for systematic hyperparameter tuning and bias-variance analysis.
