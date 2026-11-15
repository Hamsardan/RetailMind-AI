# Day 10 — Seq2Seq + Attention Demand Forecasting

## Objective

Extend Day 9's single-step LSTM demand forecasting system to support multi-step predictions using Sequence-to-Sequence (Seq2Seq) learning with attention.

## Dataset

Rossmann Store Sales dataset.

- Store: 1
- Input: 30 days of historical sales
- Output: 14 days of predicted sales
- Split: Chronological 80% training, 20% testing

## Model Architecture

The Seq2Seq architecture consists of:

1. Encoder LSTM (32 units)
2. Decoder LSTM (32 units)
3. Additive Attention
4. TimeDistributed Dense output layer

The encoder processes historical sales.

The decoder produces 14 future sales predictions.

Attention allows the decoder to assign different weights to historical time steps.

## Data Preparation

- Sort sales chronologically.
- Scale using the training-period maximum.
- Construct 30-day input windows.
- Construct 14-day target windows.
- Prevent training targets from overlapping the test period.
- Exclude windows containing missing dates or values.

## Training

- Framework: TensorFlow/Keras
- Optimizer: Adam
- Loss: Mean Squared Error
- Epochs: 20
- Batch size: 32
- Validation split: 10%

## Evaluation

Compare the Seq2Seq model against the Day 9 LSTM baseline.

The baseline uses recursive forecasting to produce multiple future predictions.

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Evaluate at:

- 7-day horizon
- 14-day horizon

## Visualizations

1. Actual vs. predicted sales
2. Seq2Seq vs. recursive LSTM predictions
3. Attention heatmap across historical days

Attention weights indicate model focus, not causality.

## Forecasting Output

The system generates daily sales predictions for the next 14 days.

The output includes:

- Forecast date
- Predicted sales
- Total forecast sales

These predictions can support procurement planning.

## Generated Artifacts

- models/demand_seq2seq_attention.keras
- models/demand_day10_scale.json
- models/demand_lstm.keras (if the Day 9 model is missing)

## Limitations

- Only Store 1 is modeled.
- No external features are included.
- Attention does not establish causality.
- Sales predictions are not equivalent to purchase-order quantities.
- Model performance must be verified after training.

## Conclusion

Day 10 extends RetailMind AI's demand forecasting module from single-step LSTM predictions to multi-step Seq2Seq forecasting with attention.

The module now supports 14-day sales forecasting, multi-horizon evaluation, and attention visualization.
