"""Tests for the delivery risk API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import HIGH_RISK_THRESHOLD, app  # noqa: E402

client = TestClient(app)

SAMPLE_ORDER = {
    "total_items": 2,
    "unique_products": 2,
    "num_sellers": 1,
    "total_price": 129.90,
    "total_freight": 18.50,
    "total_weight_g": 1400.0,
    "total_volume_cm3": 6300.0,
    "payment_value": 148.40,
    "payment_installments": 3,
    "payment_type_count": 1,
    "promised_delivery_days": 15.0,
    "shipping_limit_buffer_days": 4.0,
    "freight_share": 0.125,
    "purchase_hour": 14,
    "approval_delay_hours": 0.5,
    "customer_prior_orders": 0,
    "seller_prior_orders": 120,
    "seller_prior_late_rate_proxy": 0.06,
    "same_state_flag": False,
    "customer_repeat_flag": False,
    "customer_state": "SP",
    "primary_seller_state": "MG",
    "primary_category_name": "bed_bath_table",
    "primary_payment_type": "credit_card",
    "purchase_weekday": "Monday",
    "purchase_month_name": "March",
}


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0.0 < body["roc_auc"] < 1.0


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=SAMPLE_ORDER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["late_probability"] <= 1.0
    assert body["risk_band"] in {"low", "elevated", "high"}
    assert body["threshold_used"] == pytest.approx(HIGH_RISK_THRESHOLD)


def test_predict_rejects_invalid_payload():
    bad = dict(SAMPLE_ORDER, purchase_hour=99)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in SAMPLE_ORDER.items() if k != "total_price"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_scores_every_order():
    response = client.post("/predict/batch", json=[SAMPLE_ORDER, SAMPLE_ORDER])
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_batch_rejects_empty_list():
    assert client.post("/predict/batch", json=[]).status_code == 400


def test_unseen_category_does_not_crash():
    """OneHotEncoder(handle_unknown='ignore') must absorb new values."""
    payload = dict(SAMPLE_ORDER, primary_category_name="a_category_never_seen")
    assert client.post("/predict", json=payload).status_code == 200


def test_metrics_endpoint_exposes_roc_auc():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    assert "roc_auc" in response.json()
