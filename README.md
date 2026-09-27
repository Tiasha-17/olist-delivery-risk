# Late-Delivery Risk — Model, API & Container

A late-delivery risk model for e-commerce orders, trained on 96k Brazilian marketplace orders and served behind a FastAPI endpoint with a Docker image, tests and CI.

The point of this repo is not the notebook. It is the path from raw CSVs to a running service that returns a calibrated risk score for a single order in under 50 ms.

```bash
docker build -t delivery-risk . && docker run -p 8000:8000 delivery-risk
# then open http://localhost:8000/docs
```

---

## What it does

Olist promises every customer an estimated delivery date. About **5.3%** of orders miss it — and when they do, the average review score collapses from **4.29 to 2.57**. Those late orders are the single largest driver of bad reviews in the dataset.

This service scores an order *at checkout time*, before it ships, so operations can expedite the ones most likely to slip.

## Results

| Metric | Value | Note |
|---|---|---|
| ROC AUC (held out) | **0.740** | Chronological split, no leakage |
| Average precision | **0.114** | Base rate is 0.053 — roughly **2.1x lift** |
| Precision @ operating threshold | 15.1% | ~3x the base rate |
| Recall @ operating threshold | 11.2% | Tuned for a reviewable queue size |
| Training rows / test rows | 77,182 / 19,296 | Split at a fixed timestamp |

Average precision looks low in isolation. It should be read against the 5.3% base rate: the model roughly doubles precision over random selection on a heavily imbalanced target. The operating threshold (0.656) was chosen to keep the daily review queue small enough for a human team to actually action, which costs recall on purpose.

**Honest limitation:** an intervention simulation on the flagged queue reduced the late rate by only 0.19 percentage points. The model finds signal, but the features available at checkout do not explain most of the variance in carrier performance. Fixing that needs carrier-level and route-level data the public dataset does not contain.

## Engineering decisions worth calling out

**Chronological split, not random.** Delivery risk is a forecasting problem. A random split lets the model learn from future orders and score past ones, which inflates every metric. The split happens at a fixed timestamp; the test set sits strictly after the training set.

**All preprocessing lives inside the sklearn Pipeline.** Imputation, scaling and one-hot encoding are fitted on training data only and travel with the serialised artifact. There is no way for the API to apply a different transformation than training did.

**`handle_unknown="ignore"` on the encoder.** New seller states and product categories appear constantly in production. The encoder absorbs unseen values instead of raising. There is a test for exactly this.

**The threshold is a product decision, not a default.** 0.5 is arbitrary. 0.656 was picked from the precision/recall curve to size the review queue, and it is exposed in the API response so callers know what they are getting.

**Training is separate from analysis.** `src/train_model.py` produces a versioned artifact with embedded metrics and a UTC timestamp. The reporting pipeline and the serving path do not share state.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + which model version is loaded |
| `/predict` | POST | Score one order |
| `/predict/batch` | POST | Score up to 1,000 orders |
| `/model/metrics` | GET | Held-out evaluation metrics |
| `/docs` | GET | Auto-generated OpenAPI docs |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_items": 2, "unique_products": 2, "num_sellers": 1,
    "total_price": 129.90, "total_freight": 18.50,
    "total_weight_g": 1400, "total_volume_cm3": 6300,
    "payment_value": 148.40, "payment_installments": 3, "payment_type_count": 1,
    "promised_delivery_days": 15, "shipping_limit_buffer_days": 4,
    "freight_share": 0.125, "purchase_hour": 14, "approval_delay_hours": 0.5,
    "customer_prior_orders": 0, "seller_prior_orders": 120,
    "seller_prior_late_rate_proxy": 0.06,
    "same_state_flag": false, "customer_repeat_flag": false,
    "customer_state": "SP", "primary_seller_state": "MG",
    "primary_category_name": "bed_bath_table",
    "primary_payment_type": "credit_card",
    "purchase_weekday": "Monday", "purchase_month_name": "March"
  }'
```

```json
{
  "late_probability": 0.5428,
  "risk_band": "elevated",
  "review_recommended": false,
  "threshold_used": 0.6563
}
```

Request validation is handled by Pydantic — out-of-range hours, missing fields and wrong types are rejected with a 422 before they reach the model.

## Tech stack

Python 3.11 · scikit-learn · pandas · FastAPI · Pydantic v2 · Docker · pytest · GitHub Actions

## Repository layout

```
├── api/main.py                          # FastAPI service
├── src/
│   ├── train_model.py                   # Trains and versions the artifact
│   ├── build_delivery_performance_assets.py   # Feature + analysis pipeline
│   └── build_combined_datasets.py       # Marketing funnel joins
├── models/delivery_risk_model.joblib    # Serialised pipeline (11 KB, committed)
├── notebooks/                           # Exploratory analysis
├── reports/                             # Generated summary
├── tests/test_api.py                    # 8 tests
├── Dockerfile
└── .github/workflows/ci.yml
```

The model artifact is committed, so the API and Docker image run immediately after cloning — you do not need the 240 MB dataset unless you want to retrain.

## Running it

**API only** (no dataset needed):

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
pytest tests/ -v
```

**Full pipeline** (retraining from raw data):

```bash
# Download the Olist datasets — see data/raw/README.md
python src/build_delivery_performance_assets.py   # builds features + report
python src/train_model.py                          # retrains the artifact
```

## What I would do next

- Add carrier and route features — the biggest gap, and the most likely source of real lift
- Replace logistic regression with gradient boosting and compare honestly against this baseline
- Add drift monitoring on the input distribution, since seller mix shifts over time
- Log predictions with outcomes so the threshold can be retuned on live data

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100k orders, 2016–2018) and the companion [Marketing Funnel dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist). Raw files are gitignored; see `data/raw/README.md`.

## Licence

MIT
