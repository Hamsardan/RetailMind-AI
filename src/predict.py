import joblib
import pandas as pd
from pathlib import Path

# Locate the persisted model relative to this file
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"

# Load model once when the service is imported
model = joblib.load(MODEL_PATH)


def predict(customer_data):
    """
    Predict customer churn.

    Accepts:
        dict: A single customer's features.
        pd.DataFrame: Multiple customers' features.

    Returns:
        pd.DataFrame containing prediction and churn probability.
    """

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    required_features = list(model.feature_names_in_)

    missing_features = set(required_features) - set(customer_data.columns)
    extra_features = set(customer_data.columns) - set(required_features)

    if missing_features or extra_features:
        raise ValueError(
            f"Missing features: {missing_features}. "
            f"Unexpected features: {extra_features}."
        )

    # Maintain the exact feature order used during training
    customer_data = customer_data[required_features]

    predictions = model.predict(customer_data)
    probabilities = model.predict_proba(customer_data)[:, 1]

    return pd.DataFrame({
        "churn_prediction": ["Yes" if p == 1 else "No" for p in predictions],
        "churn_probability": probabilities
    })