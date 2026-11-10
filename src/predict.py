"""Prediction interface for the saved classical and Keras churn models."""

from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
CLASSICAL_MODEL_PATHS = {
    "day1": ROOT_DIR / "models" / "churn_pipeline.joblib",
    "day2": ROOT_DIR / "models" / "churn_day2_champion.joblib",
    "day3": ROOT_DIR / "models" / "churn_v3.joblib",
    "day4": ROOT_DIR / "models" / "churn_v4.joblib",
}
CHAMPION_FALLBACKS = tuple(
    CLASSICAL_MODEL_PATHS[name] for name in ("day4", "day3", "day2", "day1")
)
ANN_MODEL_PATH = ROOT_DIR / "models" / "churn_ann_day5.keras"
ANN_PREPROCESSOR_PATH = ROOT_DIR / "models" / "churn_ann_preprocessor.joblib"


@lru_cache(maxsize=4)
def _load_classical_model(path):
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    raise FileNotFoundError(f"Saved model not found: {path}")


def _classical_model(model_name):
    if model_name in CLASSICAL_MODEL_PATHS:
        return _load_classical_model(str(CLASSICAL_MODEL_PATHS[model_name]))
    for path in CHAMPION_FALLBACKS:
        if path.exists():
            return _load_classical_model(str(path))
    raise FileNotFoundError(
        "No saved classical churn model found. Train a notebook model first."
    )


@lru_cache(maxsize=1)
def _ann_artifacts():
    if not ANN_MODEL_PATH.exists() or not ANN_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "The Day 5 ANN artifacts are missing. Run "
            "notebooks/day5_ann.ipynb to create them."
        )
    import tensorflow as tf

    return (
        tf.keras.models.load_model(ANN_MODEL_PATH),
        joblib.load(ANN_PREPROCESSOR_PATH),
    )


def _validate_input(customer_data):
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or pandas DataFrame.")
    if customer_data.empty:
        raise ValueError("Input must contain at least one customer record.")
    return customer_data


def _check_columns(customer_data, required_features):
    missing = set(required_features) - set(customer_data.columns)
    extra = set(customer_data.columns) - set(required_features)
    if missing or extra:
        raise ValueError(
            f"Missing features: {sorted(missing)}. "
            f"Unexpected features: {sorted(extra)}."
        )
    return customer_data[list(required_features)]


def predict(customer_data, model_name="champion"):
    """Predict churn for one customer or a batch.

    Args:
        customer_data: A customer feature dictionary or a pandas DataFrame.
        model_name: ``"champion"`` (the latest saved classical model), a day
            selector (``"day1"`` through ``"day4"``), or ``"ann"`` (the Day 5
            Keras model and its saved preprocessing pipeline).

    Returns:
        A DataFrame with ``churn_prediction`` and ``churn_probability``.
    """
    customer_data = _validate_input(customer_data)

    if model_name == "champion" or model_name in CLASSICAL_MODEL_PATHS:
        model = _classical_model(model_name)
        required_features = list(model.feature_names_in_)
        ordered_data = _check_columns(customer_data, required_features)
        if "TotalCharges" in ordered_data.columns:
            ordered_data["TotalCharges"] = pd.to_numeric(
                ordered_data["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)
        predictions = model.predict(ordered_data)
        probabilities = model.predict_proba(ordered_data)[:, 1]
    elif model_name == "ann":
        ann, preprocessor = _ann_artifacts()
        required_features = list(preprocessor.feature_names_in_)
        ordered_data = _check_columns(customer_data, required_features)
        transformed = preprocessor.transform(ordered_data)
        probabilities = ann.predict(transformed, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype("int8")
    else:
        raise ValueError(
            "model_name must be 'champion', 'day1', 'day2', 'day3', 'day4', or 'ann'."
        )

    return pd.DataFrame({
        "churn_prediction": ["Yes" if value == 1 else "No" for value in predictions],
        "churn_probability": probabilities,
    }, index=customer_data.index)
