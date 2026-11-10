# Day 5 — Neural Networks and ANN (Keras)

## Objective

Learn the Keras dense-network workflow on MNIST, then benchmark a small ANN for Telco customer churn against the Phase 1 logistic champion.

## Method

- Use a dense softmax classifier for the MNIST warm-up.
- Scale numeric inputs and one-hot encode categorical inputs for churn.
- Train a 64/32-unit feed-forward network with ReLU hidden layers, dropout, a sigmoid output, balanced class weights, and early stopping on validation loss. Refit preprocessing and the network on the full training split using the selected epoch count before testing.
- Compare accuracy, precision, churn recall, and ROC-AUC on the same untouched holdout as the classical champion.

## Artifacts and decision

The ANN achieved 75.44% accuracy, 52.68% precision, 73.53% churn recall, and 83.83% ROC-AUC on the fixed holdout. Against the logistic champion, recall was lower by 4.81 percentage points and ROC-AUC was lower by 0.33 points. The classical champion remains the default prediction model.

The saved ANN can be called through `src.predict.predict(..., model_name="ann")`. The default `model_name="champion"` continues to serve the classical pipeline.
