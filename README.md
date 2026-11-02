# RetailMind AI

## An End-to-End AI-Powered Retail Intelligence Platform

RetailMind AI is a modular retail intelligence platform that integrates machine learning and data-driven technologies to help businesses understand customer behaviour, improve operational decisions and optimize business outcomes.

The platform is developed incrementally, with each module contributing to one unified system.

## Project Vision

To build an integrated AI platform for retail businesses that transforms raw business data into actionable insights, predictions and intelligent recommendations.

## Key Capabilities

The platform aims to provide:

- Customer analytics and behavioural insights.
- Predictive modelling for retail business problems.
- Intelligent recommendations and decision support.
- Data-driven operational optimization.
- Deployable machine learning services.

## Technology Stack

| Area                 | Technologies |
| -------------------- | ------------ |
| Programming Language | Python       |
| Data Processing      | Pandas       |
| Machine Learning     | Scikit-learn |
| Model Persistence    | Joblib       |

Additional technologies will be incorporated as the platform evolves.

## Current Development

### Day 1 — End-to-End ML Pipeline & ML-as-a-Service

Established the platform's initial machine learning backbone through a customer churn prediction system.

**Dataset:** Telco Customer Churn — Kaggle

**Baseline Model:** Logistic Regression

**Implementation:**

- Data ingestion, exploration and cleaning.
- Numerical feature scaling using `StandardScaler`.
- Categorical encoding using `OneHotEncoder`.
- End-to-end preprocessing and classification pipeline.
- Model evaluation and persistence using Joblib.
- Prediction service supporting individual and batch customer records.

#### Baseline Performance

Evaluated on a held-out test set of 1,409 samples.

| Metric    | Day 1 Baseline |
| --------- | -------------: |
| Accuracy  |         80.55% |
| Precision |         65.72% |
| Recall    |         55.88% |
| ROC-AUC   |         84.21% |

These metrics establish the initial performance benchmark for RetailMind AI. All subsequent model improvements will be evaluated against this baseline using consistent evaluation data and metrics.

### Day 2 — Model Comparison & Champion Selection

Trained and compared three classification algorithms against the Day 1 Logistic Regression baseline.

**Models evaluated:**

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

#### Model Comparison

Evaluated on the same 1,409-sample test set.

| Model                       | Accuracy | Precision | Recall | ROC-AUC |
| --------------------------- | -------: | --------: | -----: | ------: |
| Logistic Regression (Day 1) |   80.55% |    65.72% | 55.88% |  84.21% |
| Decision Tree               |   72.11% |    47.55% | 49.20% |  64.77% |
| Random Forest               |   78.64% |    62.54% | 48.66% |  81.85% |
| SVM                         |   79.13% |    64.18% | 48.40% |  79.02% |

**Day 2 Challenger Champion: Decision Tree**

Decision Tree achieved the highest recall (49.20%) among the three new models. However, none of the challengers outperformed the Day 1 Logistic Regression baseline.

Logistic Regression therefore remains the platform's overall champion, while Decision Tree is retained as the Day 2 challenger for future comparisons.

**Detailed results:** [Day 2 Documentation](docs/d2.md)
