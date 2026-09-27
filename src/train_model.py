"""Train the late-delivery risk model and persist it as a versioned artifact.

Separating training from the analysis pipeline lets the model be retrained,
versioned and served independently of the reporting layer.

Usage:
    python src/train_model.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TARGET = "late_flag"
TIME_COLUMN = "order_purchase_timestamp"

NUMERICAL_FEATURES = [
    "total_items",
    "unique_products",
    "num_sellers",
    "total_price",
    "total_freight",
    "total_weight_g",
    "total_volume_cm3",
    "payment_value",
    "payment_installments",
    "payment_type_count",
    "promised_delivery_days",
    "shipping_limit_buffer_days",
    "freight_share",
    "purchase_hour",
    "approval_delay_hours",
    "customer_prior_orders",
    "seller_prior_orders",
    "seller_prior_late_rate_proxy",
]

CATEGORICAL_FEATURES = [
    "same_state_flag",
    "customer_repeat_flag",
    "customer_state",
    "primary_seller_state",
    "primary_category_name",
    "primary_payment_type",
    "purchase_weekday",
    "purchase_month_name",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
    """Assemble the preprocessing + classifier pipeline.

    Everything lives inside one sklearn Pipeline so that the exact same
    transformations applied at training time are applied at inference time.
    This is what makes the artifact safe to serve.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="unknown"),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def load_training_frame() -> pd.DataFrame:
    path = PROCESSED_DIR / "delivery_order_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python src/build_delivery_performance_assets.py' first."
        )
    frame = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    return frame[
        frame["delivered_flag"]
        & frame[TARGET].notna()
        & frame["customer_state"].notna()
        & frame["primary_seller_state"].notna()
        & frame["primary_category_name"].notna()
    ].copy()


def time_based_split(frame: pd.DataFrame, train_fraction: float = 0.8):
    """Split chronologically, never randomly.

    A random split would let the model learn from future orders and score
    past ones, inflating metrics. Delivery risk is a forecasting problem,
    so the test set must sit strictly after the training set in time.
    """
    cutoff_index = int(len(frame) * train_fraction)
    cutoff = frame[TIME_COLUMN].sort_values().iloc[cutoff_index]
    train = frame[frame[TIME_COLUMN] < cutoff].copy()
    test = frame[frame[TIME_COLUMN] >= cutoff].copy()
    return train, test, cutoff


def main() -> None:
    frame = load_training_frame()
    train, test, cutoff = time_based_split(frame)

    for part in (train, test):
        for column in CATEGORICAL_FEATURES:
            part[column] = part[column].astype("string").fillna("unknown")

    model = build_pipeline()
    model.fit(train[FEATURE_COLUMNS], train[TARGET].astype(int))

    scores = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    y_true = test[TARGET].astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_late_rate": float(y_true.mean()),
        "split_cutoff": str(cutoff),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_pipeline": "median/standard-scale + constant-impute/one-hot + logistic",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS, "metrics": metrics},
        MODEL_DIR / "delivery_risk_model.joblib",
    )
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"ROC AUC          : {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Base late rate   : {metrics['test_late_rate']:.4f}")
    print(f"Saved artifact   : {MODEL_DIR / 'delivery_risk_model.joblib'}")


if __name__ == "__main__":
    main()
