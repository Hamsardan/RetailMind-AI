"""Stable, serializable feature engineering for the churn model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnInputCleaner(BaseEstimator, TransformerMixin):
    """Freeze the input cleaning needed by every serialized churn pipeline."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnInputCleaner expects a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        features = X.copy()
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"].astype(str).str.strip(), errors="coerce"
        ).fillna(0.0)
        return features

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


class ChurnFeatureEngineer(ChurnInputCleaner):
    """Add domain features while preserving the raw customer columns."""

    service_columns = (
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )

    def transform(self, X):
        features = super().transform(X)

        tenure = pd.to_numeric(features["tenure"], errors="coerce").fillna(0)
        monthly = pd.to_numeric(features["MonthlyCharges"], errors="coerce").fillna(0)
        total = features["TotalCharges"]

        features["tenure_bucket"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype(str)
        features["tenure_years"] = tenure / 12.0
        features["monthly_charge_per_tenure_month"] = total / tenure.clip(lower=1)
        expected_total = monthly * tenure
        features["total_to_expected_charge_ratio"] = np.divide(
            total,
            expected_total,
            out=np.zeros(len(features), dtype=float),
            where=expected_total.to_numpy() != 0,
        )

        enabled_services = []
        for column in self.service_columns:
            if column in features.columns:
                enabled = features[column].astype(str).eq("Yes")
                features[f"has_{column.lower()}"] = enabled.astype("int8")
                enabled_services.append(enabled.astype("int8"))
        features["enabled_service_count"] = sum(enabled_services)

        contract = features["Contract"].astype(str)
        features["is_month_to_month"] = contract.eq("Month-to-month").astype("int8")
        features["is_one_year_contract"] = contract.eq("One year").astype("int8")
        features["is_two_year_contract"] = contract.eq("Two year").astype("int8")
        features["has_internet_service"] = (
            features["InternetService"].astype(str).ne("No").astype("int8")
        )
        features["is_electronic_check"] = (
            features["PaymentMethod"].astype(str).eq("Electronic check").astype("int8")
        )
        features["has_partner_or_dependents"] = (
            features["Partner"].astype(str).eq("Yes")
            | features["Dependents"].astype(str).eq("Yes")
        ).astype("int8")
        return features

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        new_features = [
            "tenure_bucket",
            "tenure_years",
            "monthly_charge_per_tenure_month",
            "total_to_expected_charge_ratio",
            *(f"has_{column.lower()}" for column in self.service_columns),
            "enabled_service_count",
            "is_month_to_month",
            "is_one_year_contract",
            "is_two_year_contract",
            "has_internet_service",
            "is_electronic_check",
            "has_partner_or_dependents",
        ]
        return np.asarray([*input_features, *new_features], dtype=object)
