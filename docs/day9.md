# Day 9 — RNN / LSTM Demand Forecasting

## Objective

Build a first demand-forecasting model for RetailMind AI using an LSTM. The experiment uses historical daily store sales to predict sales for the following day.

## Dataset

- Dataset: Rossmann Store Sales
- Input file used by the notebook: `data/rossmann/train.csv`
- Store selected for this baseline: Store `1`
- Observations after filtering and daily preparation: `942`
- Target: `Sales`
- Time-series index: `Date`

The notebook loads the training CSV, selects Store 1, sorts the records chronologically, keeps the `Date` and `Sales` columns, sets `Date` as the index, enforces daily frequency with `asfreq("D")`, and drops missing sales records.

The experiment is deliberately limited to one store as an initial univariate baseline. Store-level metadata and other stores are not used.

## Time-series preparation

The task is framed as supervised next-step forecasting:

- Look-back window: 30 consecutive days
- Input features: one (`Sales`)
- Prediction target: sales on the day immediately after the look-back window
- Scaling: divide sales by the maximum sales value in the training portion
- Split: chronological 80:20 split, with no random shuffling

The 942 observations are split into:

| Partition       | Daily observations |
| --------------- | -----------------: |
| Training period |                753 |
| Holdout period  |                189 |
| **Total**       |            **942** |

Windows that do not span consecutive calendar dates are skipped. The resulting tensors have these shapes:

| Tensor    | Shape          |
| --------- | -------------- |
| `X_train` | `(723, 30, 1)` |
| `y_train` | `(723,)`       |
| `X_test`  | `(189, 30, 1)` |
| `y_test`  | `(189,)`       |

The 30-day history for the first holdout prediction may include observations from the end of the training period. This is appropriate for a rolling one-step-ahead evaluation because those earlier observations would already be known at prediction time.

## LSTM architecture

The model is a Keras Sequential network:

| Layer | Configuration            | Output          |
| ----- | ------------------------ | --------------- |
| Input | 30 time steps, 1 feature | `(None, 30, 1)` |
| LSTM  | 32 units                 | `(None, 32)`    |
| Dense | 1 output unit            | `(None, 1)`     |

- Total parameters: `4,385`
- Optimizer: Adam
- Loss: mean squared error (`MSE`)
- Training metric: mean absolute error (`MAE`)
- Epochs: `20`
- Batch size: `32`
- `shuffle=False`

The model is trained on the training windows. The holdout windows are passed to the trained model afterward for evaluation.

## Results

The model produced the following results on the chronological holdout:

| Metric                         |       Result |
| ------------------------------ | -----------: |
| Mean Absolute Error (MAE)      | **1,366.36** |
| Root Mean Squared Error (RMSE) | **1,870.70** |

The notebook also displays a table of actual and predicted sales values and plots actual versus predicted sales to visually inspect the fit.

A one-step-ahead forecast using the latest 30 observations produced:

- **Predicted next-day sales: 3,695.56**

These results are the output of the current Store 1 baseline. They should not be interpreted as evidence that the model generalises across all Rossmann stores.

## Saved artifact

The trained Keras model is saved to:

- `models/demand_lstm.keras`

The notebook used the training-period maximum (`max_sales`) to scale the series and to convert predictions back to the original sales scale. In the current implementation, this scaling value remains a notebook variable and is not saved as a separate artifact.

## Scope and limitations

- **Single-store experiment:** only Store 1 is used. The model is not trained across all stores.
- **Univariate input:** the model receives historical sales only. Store opening status, promotions, weekdays, and holiday indicators are not used as model features.
- **Holdout versus official test data:** `X_test` and `y_test` are created from the final 20% of the supplied training series. They are an internal chronological holdout, not the official Rossmann `test.csv`.
- **No official test predictions:** the current notebook does not load the competition's unlabeled `test.csv` or generate a Kaggle submission.
- **No recursive multi-step forecast:** the final printed prediction is a one-step-ahead forecast, not a forecast across the full official test horizon.
- **Baseline evaluation:** no seasonal-naive or other baseline model is reported, so the MAE and RMSE should be interpreted on their own rather than as proof of improvement over a simpler forecasting method.

## Conclusion

Day 9 establishes a working end-to-end univariate LSTM baseline for next-day sales forecasting on Rossmann Store 1. The notebook prepares daily sales data, creates 30-day supervised windows, trains a 32-unit LSTM, evaluates it with MAE and RMSE, visualises the holdout predictions, and saves the trained model. Extending the experiment to all stores, adding calendar and promotion features, and evaluating against the official competition test data remain future improvements.
