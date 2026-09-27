"""FastAPI service exposing the late-delivery risk model.

Run locally:
    uvicorn api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "delivery_risk_model.joblib"

# Orders above this score go into the operations review queue. Chosen from the
# precision/recall trade-off in reports/delivery_performance_summary.md rather
# than left at the default 0.5.
HIGH_RISK_THRESHOLD = 0.6563

app = FastAPI(
    title="Olist Delivery Risk API",
    description=(
        "Scores an e-commerce order for the probability that it will be "
        "delivered later than promised."
    ),
    version="1.0.0",
)

_artifact: dict | None = None


def get_artifact() -> dict:
    """Load the model once and reuse it across requests."""
    global _artifact
    if _artifact is None:
        if not ARTIFACT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run 'python src/train_model.py' first.",
            )
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


class OrderFeatures(BaseModel):
    """One order, described by the features the model was trained on."""

    total_items: int = Field(..., ge=1, examples=[2])
    unique_products: int = Field(..., ge=1, examples=[2])
    num_sellers: int = Field(..., ge=1, examples=[1])
    total_price: float = Field(..., ge=0, examples=[129.90])
    total_freight: float = Field(..., ge=0, examples=[18.50])
    total_weight_g: float = Field(..., ge=0, examples=[1400.0])
    total_volume_cm3: float = Field(..., ge=0, examples=[6300.0])
    payment_value: float = Field(..., ge=0, examples=[148.40])
    payment_installments: int = Field(..., ge=0, examples=[3])
    payment_type_count: int = Field(..., ge=1, examples=[1])
    promised_delivery_days: float = Field(..., examples=[15.0])
    shipping_limit_buffer_days: float = Field(..., examples=[4.0])
    freight_share: float = Field(..., ge=0, le=1, examples=[0.125])
    purchase_hour: int = Field(..., ge=0, le=23, examples=[14])
    approval_delay_hours: float = Field(..., ge=0, examples=[0.5])
    customer_prior_orders: int = Field(..., ge=0, examples=[0])
    seller_prior_orders: int = Field(..., ge=0, examples=[120])
    seller_prior_late_rate_proxy: float = Field(..., ge=0, le=1, examples=[0.06])

    same_state_flag: bool = Field(..., examples=[False])
    customer_repeat_flag: bool = Field(..., examples=[False])
    customer_state: str = Field(..., examples=["SP"])
    primary_seller_state: str = Field(..., examples=["MG"])
    primary_category_name: str = Field(..., examples=["bed_bath_table"])
    primary_payment_type: str = Field(..., examples=["credit_card"])
    purchase_weekday: str = Field(..., examples=["Monday"])
    purchase_month_name: str = Field(..., examples=["March"])


class PredictionResponse(BaseModel):
    late_probability: float = Field(..., description="Probability the order arrives late")
    risk_band: Literal["low", "elevated", "high"]
    review_recommended: bool
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    roc_auc: float | None = None
    trained_at: str | None = None


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe that also surfaces which model version is loaded."""
    if not ARTIFACT_PATH.exists():
        return HealthResponse(status="degraded", model_loaded=False)
    metrics = get_artifact()["metrics"]
    return HealthResponse(
        status="ok",
        model_loaded=True,
        roc_auc=round(metrics["roc_auc"], 4),
        trained_at=metrics.get("trained_at"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(order: OrderFeatures) -> PredictionResponse:
    """Score a single order for late-delivery risk."""
    artifact = get_artifact()
    frame = pd.DataFrame([order.model_dump()])[artifact["feature_columns"]]

    probability = float(artifact["model"].predict_proba(frame)[0, 1])

    if probability >= HIGH_RISK_THRESHOLD:
        band = "high"
    elif probability >= 0.25:
        band = "elevated"
    else:
        band = "low"

    return PredictionResponse(
        late_probability=round(probability, 4),
        risk_band=band,
        review_recommended=probability >= HIGH_RISK_THRESHOLD,
        threshold_used=HIGH_RISK_THRESHOLD,
    )


@app.post("/predict/batch", tags=["scoring"])
def predict_batch(orders: list[OrderFeatures]) -> dict:
    """Score up to 1,000 orders in one call."""
    if not orders:
        raise HTTPException(status_code=400, detail="Provide at least one order.")
    if len(orders) > 1000:
        raise HTTPException(status_code=413, detail="Maximum 1,000 orders per request.")

    artifact = get_artifact()
    frame = pd.DataFrame([o.model_dump() for o in orders])[artifact["feature_columns"]]
    probabilities = artifact["model"].predict_proba(frame)[:, 1]

    return {
        "count": len(orders),
        "threshold_used": HIGH_RISK_THRESHOLD,
        "flagged_for_review": int((probabilities >= HIGH_RISK_THRESHOLD).sum()),
        "predictions": [
            {
                "index": i,
                "late_probability": round(float(p), 4),
                "review_recommended": bool(p >= HIGH_RISK_THRESHOLD),
            }
            for i, p in enumerate(probabilities)
        ],
    }


@app.get("/model/metrics", tags=["ops"])
def model_metrics() -> dict:
    """Return the held-out evaluation metrics for the loaded model."""
    metrics_path = ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    return json.loads(metrics_path.read_text())
