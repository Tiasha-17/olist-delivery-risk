from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "orders": pd.read_csv(
            RAW_DIR / "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "order_items": pd.read_csv(
            RAW_DIR / "olist_order_items_dataset.csv",
            parse_dates=["shipping_limit_date"],
        ),
        "reviews": pd.read_csv(
            RAW_DIR / "olist_order_reviews_dataset.csv",
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        ),
        "customers": pd.read_csv(RAW_DIR / "olist_customers_dataset.csv"),
        "sellers": pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv"),
        "products": pd.read_csv(RAW_DIR / "olist_products_dataset.csv"),
        "payments": pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv"),
        "category_translation": pd.read_csv(RAW_DIR / "product_category_name_translation.csv"),
    }


def days_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 86400


def hours_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 3600


def dominant_value(
    frame: pd.DataFrame,
    group_cols: str | list[str],
    value_col: str,
    weight_col: str,
    output_col: str,
) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    weighted = (
        frame.assign(
            _dominant_value=frame[value_col].fillna("unknown"),
            _dominant_weight=frame[weight_col].fillna(0),
        )
        .groupby(group_cols + ["_dominant_value"], as_index=False)
        .agg(total_weight=("_dominant_weight", "sum"))
        .sort_values(group_cols + ["total_weight", "_dominant_value"], ascending=[True] * len(group_cols) + [False, True])
        .drop_duplicates(group_cols)
        .rename(columns={"_dominant_value": output_col})
    )
    return weighted[group_cols + [output_col]]


def build_order_items_enriched(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    products_enriched = products.merge(
        category_translation,
        on="product_category_name",
        how="left",
    )
    products_enriched["product_category_name_english"] = products_enriched[
        "product_category_name_english"
    ].fillna(products_enriched["product_category_name"]).fillna("unknown")

    for column in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        median_value = products_enriched[column].median()
        products_enriched[column] = products_enriched[column].fillna(median_value)

    products_enriched["product_volume_cm3"] = (
        products_enriched["product_length_cm"]
        * products_enriched["product_height_cm"]
        * products_enriched["product_width_cm"]
    )

    items = (
        order_items.merge(
            products_enriched[
                [
                    "product_id",
                    "product_category_name_english",
                    "product_weight_g",
                    "product_volume_cm3",
                ]
            ],
            on="product_id",
            how="left",
        )
        .merge(
            sellers[["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left",
        )
    )
    items["product_category_name_english"] = items["product_category_name_english"].fillna("unknown")
    return items


def build_delivery_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    reviews: pd.DataFrame,
    payments: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    reviews_summary = (
        reviews.assign(
            has_review_comment=reviews["review_comment_title"].notna() | reviews["review_comment_message"].notna()
        )
        .groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "nunique"),
            has_review_comment=("has_review_comment", "max"),
        )
    )

    payments_summary = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type_count=("payment_type", "nunique"),
        )
        .merge(
            dominant_value(payments, "order_id", "payment_type", "payment_value", "primary_payment_type"),
            on="order_id",
            how="left",
        )
    )

    order_item_summary = (
        items.groupby("order_id", as_index=False)
        .agg(
            total_items=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            num_sellers=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            total_weight_g=("product_weight_g", "sum"),
            total_volume_cm3=("product_volume_cm3", "sum"),
            min_shipping_limit_date=("shipping_limit_date", "min"),
            max_shipping_limit_date=("shipping_limit_date", "max"),
        )
        .merge(
            dominant_value(items, "order_id", "seller_id", "price", "primary_seller_id"),
            on="order_id",
            how="left",
        )
        .merge(
            dominant_value(
                items,
                "order_id",
                "product_category_name_english",
                "price",
                "primary_category_name",
            ),
            on="order_id",
            how="left",
        )
    )

    primary_seller_region = (
        items[["seller_id", "seller_city", "seller_state"]]
        .drop_duplicates("seller_id")
        .rename(
            columns={
                "seller_id": "primary_seller_id",
                "seller_city": "primary_seller_city",
                "seller_state": "primary_seller_state",
            }
        )
    )

    delivery_orders = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(reviews_summary, on="order_id", how="left")
        .merge(payments_summary, on="order_id", how="left")
        .merge(order_item_summary, on="order_id", how="left")
        .merge(primary_seller_region, on="primary_seller_id", how="left")
    )

    delivery_orders["approval_delay_hours"] = hours_between(
        delivery_orders["order_approved_at"],
        delivery_orders["order_purchase_timestamp"],
    )
    approval_anchor = delivery_orders["order_approved_at"].fillna(delivery_orders["order_purchase_timestamp"])
    delivery_orders["seller_processing_days"] = days_between(
        delivery_orders["order_delivered_carrier_date"],
        approval_anchor,
    )
    delivery_orders["carrier_delivery_days"] = days_between(
        delivery_orders["order_delivered_customer_date"],
        delivery_orders["order_delivered_carrier_date"],
    )
    delivery_orders["promised_delivery_days"] = days_between(
        delivery_orders["order_estimated_delivery_date"],
        delivery_orders["order_purchase_timestamp"],
    )
    delivery_orders["actual_delivery_days"] = days_between(
        delivery_orders["order_delivered_customer_date"],
        delivery_orders["order_purchase_timestamp"],
    )
    delivery_orders["delay_days"] = days_between(
        delivery_orders["order_delivered_customer_date"],
        delivery_orders["order_estimated_delivery_date"],
    )
    delivery_orders["shipping_limit_buffer_days"] = days_between(
        delivery_orders["min_shipping_limit_date"],
        delivery_orders["order_purchase_timestamp"],
    )

    delivery_orders["delivered_flag"] = delivery_orders["order_status"].eq("delivered")
    delivery_orders["canceled_flag"] = delivery_orders["order_status"].eq("canceled")
    delivery_orders["on_time_flag"] = delivery_orders["delay_days"].le(0).astype("boolean")
    delivery_orders["late_flag"] = delivery_orders["delay_days"].gt(0).astype("boolean")
    delivery_orders["severe_late_flag"] = delivery_orders["delay_days"].gt(3).astype("boolean")
    delivery_orders["same_state_flag"] = delivery_orders["primary_seller_state"].eq(
        delivery_orders["customer_state"]
    )

    for column in ["on_time_flag", "late_flag", "severe_late_flag"]:
        delivery_orders.loc[~delivery_orders["delivered_flag"], column] = pd.NA

    delivery_orders["freight_share"] = (
        delivery_orders["total_freight"] / delivery_orders["total_price"]
    ).replace([np.inf, -np.inf], np.nan)
    delivery_orders["purchase_month"] = (
        delivery_orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )
    delivery_orders["purchase_weekday"] = delivery_orders["order_purchase_timestamp"].dt.day_name()
    delivery_orders["purchase_hour"] = delivery_orders["order_purchase_timestamp"].dt.hour
    delivery_orders["purchase_month_name"] = delivery_orders["order_purchase_timestamp"].dt.month_name()

    conditions = [
        ~delivery_orders["delivered_flag"],
        delivery_orders["delay_days"].gt(3),
        delivery_orders["delay_days"].gt(0),
        delivery_orders["delay_days"].le(0),
    ]
    labels = ["not_delivered", "late_4plus_days", "late_1to3_days", "on_time_or_early"]
    delivery_orders["delivery_experience"] = np.select(conditions, labels, default="unknown")

    delivery_orders = delivery_orders.sort_values("order_purchase_timestamp").reset_index(drop=True)
    delivery_orders["customer_prior_orders"] = delivery_orders.groupby("customer_unique_id").cumcount()
    delivery_orders["seller_prior_orders"] = delivery_orders.groupby("primary_seller_id").cumcount()
    delivered_int = delivery_orders["delivered_flag"].astype(int)
    late_int = delivery_orders["late_flag"].fillna(False).astype(int)
    seller_cum_delivered = delivered_int.groupby(delivery_orders["primary_seller_id"]).cumsum() - delivered_int
    seller_cum_late = late_int.groupby(delivery_orders["primary_seller_id"]).cumsum() - late_int
    delivery_orders["seller_prior_late_rate_proxy"] = (
        seller_cum_late / seller_cum_delivered.replace(0, np.nan)
    )
    delivery_orders["customer_repeat_flag"] = delivery_orders["customer_prior_orders"].gt(0)

    return delivery_orders


def build_seller_order_delivery(items: pd.DataFrame, delivery_orders: pd.DataFrame) -> pd.DataFrame:
    seller_order_delivery = (
        items.groupby(["order_id", "seller_id"], as_index=False)
        .agg(
            seller_item_count=("order_item_id", "count"),
            seller_unique_products=("product_id", "nunique"),
            seller_revenue=("price", "sum"),
            seller_freight=("freight_value", "sum"),
            seller_weight_g=("product_weight_g", "sum"),
            seller_volume_cm3=("product_volume_cm3", "sum"),
            seller_city=("seller_city", "first"),
            seller_state=("seller_state", "first"),
        )
        .merge(
            dominant_value(
                items,
                ["order_id", "seller_id"],
                "product_category_name_english",
                "price",
                "seller_primary_category",
            ),
            on=["order_id", "seller_id"],
            how="left",
        )
        .merge(
            delivery_orders[
                [
                    "order_id",
                    "customer_id",
                    "customer_unique_id",
                    "customer_city",
                    "customer_state",
                    "order_status",
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "review_score",
                    "payment_value",
                    "payment_installments",
                    "primary_payment_type",
                    "approval_delay_hours",
                    "seller_processing_days",
                    "carrier_delivery_days",
                    "promised_delivery_days",
                    "actual_delivery_days",
                    "delay_days",
                    "delivered_flag",
                    "canceled_flag",
                    "on_time_flag",
                    "late_flag",
                    "severe_late_flag",
                    "delivery_experience",
                ]
            ],
            on="order_id",
            how="left",
        )
    )

    seller_order_delivery["seller_revenue_share"] = (
        seller_order_delivery["seller_revenue"] / seller_order_delivery["payment_value"]
    ).replace([np.inf, -np.inf], np.nan)
    seller_order_delivery["same_state_flag"] = seller_order_delivery["seller_state"].eq(
        seller_order_delivery["customer_state"]
    )
    return seller_order_delivery


def build_seller_reliability(seller_order_delivery: pd.DataFrame) -> pd.DataFrame:
    delivered_seller_orders = seller_order_delivery[seller_order_delivery["delivered_flag"]].copy()

    repeat_customers = (
        seller_order_delivery.groupby(["seller_id", "customer_unique_id"], as_index=False)
        .agg(customer_orders=("order_id", "nunique"))
        .assign(repeat_customer=lambda frame: frame["customer_orders"].gt(1))
        .groupby("seller_id", as_index=False)
        .agg(repeat_customer_rate=("repeat_customer", "mean"))
    )

    seller_reliability = (
        seller_order_delivery.groupby("seller_id", as_index=False)
        .agg(
            seller_city=("seller_city", "first"),
            seller_state=("seller_state", "first"),
            total_orders=("order_id", "nunique"),
            total_customers=("customer_unique_id", "nunique"),
            total_revenue=("seller_revenue", "sum"),
            total_freight=("seller_freight", "sum"),
            delivered_orders=("delivered_flag", "sum"),
            canceled_orders=("canceled_flag", "sum"),
            avg_review_score=("review_score", "mean"),
            avg_order_value=("seller_revenue", "mean"),
            same_state_rate=("same_state_flag", "mean"),
            first_order_at=("order_purchase_timestamp", "min"),
            last_order_at=("order_purchase_timestamp", "max"),
        )
        .merge(
            delivered_seller_orders.groupby("seller_id", as_index=False).agg(
                on_time_rate=("on_time_flag", "mean"),
                late_order_rate=("late_flag", "mean"),
                severe_late_rate=("severe_late_flag", "mean"),
                avg_delay_days=("delay_days", "mean"),
                median_delay_days=("delay_days", "median"),
                avg_approval_delay_hours=("approval_delay_hours", "mean"),
                avg_seller_processing_days=("seller_processing_days", "mean"),
                avg_carrier_delivery_days=("carrier_delivery_days", "mean"),
                avg_actual_delivery_days=("actual_delivery_days", "mean"),
            ),
            on="seller_id",
            how="left",
        )
        .merge(repeat_customers, on="seller_id", how="left")
    )

    seller_reliability["cancellation_rate"] = (
        seller_reliability["canceled_orders"] / seller_reliability["total_orders"]
    ).replace([np.inf, -np.inf], np.nan)
    seller_reliability["active_days"] = (
        seller_reliability["last_order_at"] - seller_reliability["first_order_at"]
    ).dt.days.fillna(0)
    seller_reliability["active_months"] = (seller_reliability["active_days"] / 30).clip(lower=1)
    seller_reliability["orders_per_month"] = (
        seller_reliability["total_orders"] / seller_reliability["active_months"]
    )

    for column in [
        "on_time_rate",
        "avg_review_score",
        "cancellation_rate",
        "avg_seller_processing_days",
    ]:
        seller_reliability[column] = pd.to_numeric(seller_reliability[column], errors="coerce")

    global_on_time = delivered_seller_orders["on_time_flag"].mean()
    global_review = seller_order_delivery["review_score"].mean()
    processing_cap = delivered_seller_orders["seller_processing_days"].quantile(0.95)
    processing_cap = 1 if pd.isna(processing_cap) or processing_cap <= 0 else processing_cap

    seller_reliability["reliability_score"] = 100 * (
        0.35 * seller_reliability["on_time_rate"].fillna(global_on_time).clip(0, 1)
        + 0.25 * (seller_reliability["avg_review_score"].fillna(global_review) / 5).clip(0, 1)
        + 0.15 * (1 - seller_reliability["cancellation_rate"].fillna(0).clip(0, 1))
        + 0.15
        * (1 - seller_reliability["avg_seller_processing_days"].fillna(processing_cap) / processing_cap).clip(0, 1)
        + 0.10 * (np.log1p(seller_reliability["delivered_orders"]) / np.log1p(50)).clip(0, 1)
    )

    seller_reliability["eligible_for_ranking"] = seller_reliability["delivered_orders"].ge(10)

    baseline_approval = delivered_seller_orders.loc[
        delivered_seller_orders["on_time_flag"].eq(True),
        "approval_delay_hours",
    ].mean()
    baseline_seller = delivered_seller_orders.loc[
        delivered_seller_orders["on_time_flag"].eq(True),
        "seller_processing_days",
    ].mean()
    baseline_carrier = delivered_seller_orders.loc[
        delivered_seller_orders["on_time_flag"].eq(True),
        "carrier_delivery_days",
    ].mean()

    def dominant_root_cause(row: pd.Series) -> str:
        if row["cancellation_rate"] >= 0.08:
            return "cancellations"

        stage_excess = {
            "approval_latency": row["avg_approval_delay_hours"] - baseline_approval,
            "seller_handoff": row["avg_seller_processing_days"] - baseline_seller,
            "carrier_transit": row["avg_carrier_delivery_days"] - baseline_carrier,
        }
        top_stage = max(stage_excess, key=stage_excess.get)
        if pd.isna(stage_excess[top_stage]) or stage_excess[top_stage] <= 0.5:
            return "balanced"
        return top_stage

    seller_reliability["dominant_root_cause"] = seller_reliability.apply(dominant_root_cause, axis=1)

    seller_reliability["reliability_tier"] = pd.cut(
        seller_reliability["reliability_score"],
        bins=[-np.inf, 55, 70, 85, np.inf],
        labels=["critical", "watchlist", "stable", "elite"],
    ).astype(str)

    seller_reliability["reliability_rank"] = (
        seller_reliability.loc[seller_reliability["eligible_for_ranking"], "reliability_score"]
        .rank(method="dense", ascending=False)
    )

    return seller_reliability.sort_values(
        ["eligible_for_ranking", "reliability_score", "delivered_orders"],
        ascending=[False, False, False],
    )


def build_customer_retention(delivery_orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_orders = delivery_orders[
        delivery_orders["customer_unique_id"].notna() & delivery_orders["order_purchase_timestamp"].notna()
    ][
        [
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "payment_value",
            "delivery_experience",
        ]
    ].copy()

    customer_orders["purchase_month"] = (
        customer_orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )
    first_orders = (
        customer_orders.sort_values("order_purchase_timestamp")
        .drop_duplicates("customer_unique_id")
        .rename(
            columns={
                "purchase_month": "cohort_month",
                "delivery_experience": "first_order_delivery_segment",
            }
        )[
            ["customer_unique_id", "cohort_month", "first_order_delivery_segment"]
        ]
    )

    customer_orders = customer_orders.merge(first_orders, on="customer_unique_id", how="left")
    customer_orders["months_since_first"] = (
        (customer_orders["purchase_month"].dt.year - customer_orders["cohort_month"].dt.year) * 12
        + (customer_orders["purchase_month"].dt.month - customer_orders["cohort_month"].dt.month)
    )

    cohort_size = (
        first_orders.groupby(["cohort_month", "first_order_delivery_segment"], as_index=False)
        .agg(cohort_size=("customer_unique_id", "nunique"))
    )

    retention = (
        customer_orders.groupby(
            ["cohort_month", "first_order_delivery_segment", "months_since_first"],
            as_index=False,
        )
        .agg(
            active_customers=("customer_unique_id", "nunique"),
            total_orders=("order_id", "nunique"),
            avg_order_value=("payment_value", "mean"),
        )
        .merge(cohort_size, on=["cohort_month", "first_order_delivery_segment"], how="left")
    )
    retention["retention_rate"] = retention["active_customers"] / retention["cohort_size"]

    repeat_flags = (
        customer_orders.groupby("customer_unique_id", as_index=False)
        .agg(max_month=("months_since_first", "max"), total_orders=("order_id", "nunique"))
        .assign(repeat_customer=lambda frame: frame["max_month"].gt(0))
        .merge(first_orders, on="customer_unique_id", how="left")
    )

    retention_summary = (
        repeat_flags.groupby("first_order_delivery_segment", as_index=False)
        .agg(
            customers=("customer_unique_id", "nunique"),
            repeat_customers=("repeat_customer", "sum"),
            avg_orders_per_customer=("total_orders", "mean"),
        )
    )
    retention_summary["repeat_rate"] = retention_summary["repeat_customers"] / retention_summary["customers"]

    return retention, retention_summary.sort_values("customers", ascending=False)


def build_driver_summaries(delivery_orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delivered = delivery_orders[delivery_orders["delivered_flag"]].copy()

    delivered["weight_bucket"] = pd.cut(
        delivered["total_weight_g"],
        bins=[-np.inf, 500, 2000, 5000, 20000, np.inf],
        labels=["<=0.5kg", "0.5-2kg", "2-5kg", "5-20kg", "20kg+"],
    ).astype(str)
    delivered["order_size_bucket"] = pd.cut(
        delivered["total_items"],
        bins=[-np.inf, 1, 2, 5, np.inf],
        labels=["1 item", "2 items", "3-5 items", "6+ items"],
    ).astype(str)
    delivered["promised_window_bucket"] = pd.cut(
        delivered["promised_delivery_days"],
        bins=[-np.inf, 5, 10, 15, 20, np.inf],
        labels=["<=5 days", "6-10 days", "11-15 days", "16-20 days", "20+ days"],
    ).astype(str)
    delivered["same_state_bucket"] = np.where(
        delivered["same_state_flag"],
        "same_state",
        "cross_state",
    )

    def summarize(segment_type: str, column: str) -> pd.DataFrame:
        frame = (
            delivered.assign(segment_value=delivered[column].fillna("unknown").astype(str))
            .groupby("segment_value", as_index=False)
            .agg(
                delivered_orders=("order_id", "nunique"),
                on_time_rate=("on_time_flag", "mean"),
                late_rate=("late_flag", "mean"),
                avg_review_score=("review_score", "mean"),
                avg_delay_days=("delay_days", "mean"),
                avg_promised_delivery_days=("promised_delivery_days", "mean"),
                avg_actual_delivery_days=("actual_delivery_days", "mean"),
                total_payment_value=("payment_value", "sum"),
            )
        )
        frame["segment_type"] = segment_type
        return frame

    segments = [
        summarize("customer_state", "customer_state"),
        summarize("seller_state", "primary_seller_state"),
        summarize("primary_category", "primary_category_name"),
        summarize("route_type", "same_state_bucket"),
        summarize("weight_bucket", "weight_bucket"),
        summarize("order_size_bucket", "order_size_bucket"),
        summarize("promised_window_bucket", "promised_window_bucket"),
    ]
    driver_summary = pd.concat(segments, ignore_index=True)

    regional_summary = pd.concat(
        [
            summarize("customer_state", "customer_state"),
            summarize("seller_state", "primary_seller_state"),
        ],
        ignore_index=True,
    )

    stage_benchmarks = (
        delivered.groupby("delivery_experience", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            avg_review_score=("review_score", "mean"),
            avg_approval_delay_hours=("approval_delay_hours", "mean"),
            avg_seller_processing_days=("seller_processing_days", "mean"),
            avg_carrier_delivery_days=("carrier_delivery_days", "mean"),
            avg_promised_delivery_days=("promised_delivery_days", "mean"),
            avg_actual_delivery_days=("actual_delivery_days", "mean"),
        )
    )

    return driver_summary, regional_summary, stage_benchmarks


def train_delivery_risk_model(delivery_orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frame = delivery_orders[
        delivery_orders["delivered_flag"]
        & delivery_orders["late_flag"].notna()
        & delivery_orders["customer_state"].notna()
        & delivery_orders["primary_seller_state"].notna()
        & delivery_orders["primary_category_name"].notna()
    ].copy()

    model_frame["purchase_month_name"] = model_frame["purchase_month_name"].fillna("unknown")
    model_frame["purchase_weekday"] = model_frame["purchase_weekday"].fillna("unknown")
    model_frame["primary_payment_type"] = model_frame["primary_payment_type"].fillna("unknown")

    feature_columns = [
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
        "same_state_flag",
        "purchase_hour",
        "approval_delay_hours",
        "customer_prior_orders",
        "seller_prior_orders",
        "seller_prior_late_rate_proxy",
        "customer_repeat_flag",
        "customer_state",
        "primary_seller_state",
        "primary_category_name",
        "primary_payment_type",
        "purchase_weekday",
        "purchase_month_name",
    ]

    numerical_features = [
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
    categorical_features = [
        "same_state_flag",
        "customer_repeat_flag",
        "customer_state",
        "primary_seller_state",
        "primary_category_name",
        "primary_payment_type",
        "purchase_weekday",
        "purchase_month_name",
    ]

    cutoff_index = int(len(model_frame) * 0.8)
    cutoff_timestamp = model_frame["order_purchase_timestamp"].sort_values().iloc[cutoff_index]
    train_mask = model_frame["order_purchase_timestamp"].lt(cutoff_timestamp)
    test_mask = ~train_mask

    train = model_frame.loc[train_mask].copy()
    test = model_frame.loc[test_mask].copy()

    for frame in (train, test):
        for column in categorical_features:
            frame[column] = frame[column].astype("string").fillna("unknown")

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
                numerical_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        # strategy="constant" is dtype-safe across sklearn
                        # versions; "most_frequent" breaks on object columns
                        # in scikit-learn >= 1.6.
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="unknown"),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(train[feature_columns], train["late_flag"].astype(int))

    train_scores = model.predict_proba(train[feature_columns])[:, 1]
    test_scores = model.predict_proba(test[feature_columns])[:, 1]

    threshold = float(np.quantile(train_scores, 0.85))
    test["predicted_late_risk"] = test_scores
    test["high_risk_flag"] = test["predicted_late_risk"].ge(threshold)
    test["actual_late_flag"] = test["late_flag"].astype(int)

    model_metrics = pd.DataFrame(
        [
            {"metric": "train_rows", "value": len(train)},
            {"metric": "test_rows", "value": len(test)},
            {"metric": "test_late_rate", "value": test["actual_late_flag"].mean()},
            {"metric": "roc_auc", "value": roc_auc_score(test["actual_late_flag"], test_scores)},
            {
                "metric": "average_precision",
                "value": average_precision_score(test["actual_late_flag"], test_scores),
            },
            {
                "metric": "high_risk_precision",
                "value": precision_score(test["actual_late_flag"], test["high_risk_flag"], zero_division=0),
            },
            {
                "metric": "high_risk_recall",
                "value": recall_score(test["actual_late_flag"], test["high_risk_flag"], zero_division=0),
            },
            {"metric": "high_risk_threshold", "value": threshold},
        ]
    )

    predictions = test[
        [
            "order_id",
            "order_purchase_timestamp",
            "customer_state",
            "primary_seller_state",
            "primary_category_name",
            "promised_delivery_days",
            "delay_days",
            "actual_late_flag",
            "predicted_late_risk",
            "high_risk_flag",
        ]
    ].sort_values("predicted_late_risk", ascending=False)

    return predictions, model_metrics


def simulate_interventions(predictions: pd.DataFrame) -> pd.DataFrame:
    high_risk_mask = predictions["high_risk_flag"]

    def scenario_summary(
        scenario: str,
        actual_reduction_days: float = 0,
        promise_extension_days: float = 0,
    ) -> dict[str, float | str]:
        adjusted_delay = predictions["delay_days"].copy()
        adjusted_delay.loc[high_risk_mask] = (
            adjusted_delay.loc[high_risk_mask] - actual_reduction_days - promise_extension_days
        )
        simulated_late = adjusted_delay.gt(0).astype(int)
        baseline_late_orders = predictions["actual_late_flag"].sum()
        late_orders_after = simulated_late.sum()
        flagged_orders = int(high_risk_mask.sum())

        return {
            "scenario": scenario,
            "flagged_orders": flagged_orders,
            "flagged_share": flagged_orders / len(predictions),
            "baseline_late_rate": predictions["actual_late_flag"].mean(),
            "simulated_late_rate": simulated_late.mean(),
            "late_orders_avoided": baseline_late_orders - late_orders_after,
            "reduction_pp": (predictions["actual_late_flag"].mean() - simulated_late.mean()) * 100,
        }

    scenarios = [
        scenario_summary("baseline_no_change"),
        scenario_summary("high_risk_expedite_1_day", actual_reduction_days=1),
        scenario_summary("high_risk_expedite_2_days", actual_reduction_days=2),
        scenario_summary("high_risk_promise_buffer_plus_2_days", promise_extension_days=2),
        scenario_summary(
            "high_risk_expedite_1_day_plus_1_day_buffer",
            actual_reduction_days=1,
            promise_extension_days=1,
        ),
    ]
    return pd.DataFrame(scenarios)


def build_seller_recommendations(seller_reliability: pd.DataFrame) -> pd.DataFrame:
    def priority(row: pd.Series) -> str:
        reliability_score = row["reliability_score"]
        on_time_rate = row["on_time_rate"]
        late_order_rate = row["late_order_rate"]

        if pd.isna(reliability_score):
            return "medium"
        if reliability_score < 55 or (pd.notna(on_time_rate) and on_time_rate < 0.8):
            return "critical"
        if reliability_score < 70 or (pd.notna(late_order_rate) and late_order_rate > 0.15):
            return "high"
        if reliability_score < 85:
            return "medium"
        return "low"

    def recommendation(row: pd.Series) -> str:
        if row["dominant_root_cause"] == "cancellations":
            return "Improve stock accuracy and catalog availability; cancellations are damaging trust and reliability."
        if row["dominant_root_cause"] == "seller_handoff":
            return "Reduce seller handoff time by staging inventory earlier and tightening pickup SLAs."
        if row["dominant_root_cause"] == "carrier_transit":
            return "Review carrier mix and destination-state routing to shorten last-mile delivery times."
        if row["dominant_root_cause"] == "approval_latency":
            return "Streamline payment approval and fraud-review steps before fulfillment begins."
        return "Maintain current operations and monitor review score, on-time rate, and cancellation drift."

    recommendations = seller_reliability.copy()
    recommendations["priority"] = recommendations.apply(priority, axis=1)
    recommendations["recommended_action"] = recommendations.apply(recommendation, axis=1)
    recommendations["monitor_metric"] = np.select(
        [
            recommendations["dominant_root_cause"].eq("cancellations"),
            recommendations["dominant_root_cause"].eq("seller_handoff"),
            recommendations["dominant_root_cause"].eq("carrier_transit"),
            recommendations["dominant_root_cause"].eq("approval_latency"),
        ],
        [
            "cancellation_rate",
            "avg_seller_processing_days",
            "avg_carrier_delivery_days",
            "avg_approval_delay_hours",
        ],
        default="reliability_score",
    )

    return recommendations[
        [
            "seller_id",
            "seller_state",
            "delivered_orders",
            "total_revenue",
            "on_time_rate",
            "avg_review_score",
            "cancellation_rate",
            "reliability_score",
            "reliability_tier",
            "dominant_root_cause",
            "priority",
            "monitor_metric",
            "recommended_action",
        ]
    ].sort_values(["priority", "reliability_score", "delivered_orders"], ascending=[True, True, False])


def write_summary_report(
    delivery_orders: pd.DataFrame,
    regional_summary: pd.DataFrame,
    stage_benchmarks: pd.DataFrame,
    seller_reliability: pd.DataFrame,
    retention_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    intervention_simulation: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    delivered = delivery_orders[delivery_orders["delivered_flag"]].copy()
    on_time = delivered["on_time_flag"].mean()
    review_on_time = delivered.loc[delivered["on_time_flag"].eq(True), "review_score"].mean()
    review_late = delivered.loc[delivered["late_flag"].eq(True), "review_score"].mean()

    customer_states = regional_summary[
        (regional_summary["segment_type"] == "customer_state")
        & (regional_summary["delivered_orders"] >= 500)
    ].sort_values("on_time_rate", ascending=False)

    top_regions = customer_states.head(3)[["segment_value", "on_time_rate", "delivered_orders"]]
    bottom_regions = customer_states.tail(3)[["segment_value", "on_time_rate", "delivered_orders"]]

    top_sellers = seller_reliability[
        seller_reliability["eligible_for_ranking"]
    ].head(5)[["seller_id", "reliability_score", "on_time_rate", "delivered_orders"]]

    repeat_rates = retention_summary[["first_order_delivery_segment", "repeat_rate", "customers"]]
    metrics = dict(zip(model_metrics["metric"], model_metrics["value"]))
    best_intervention = intervention_simulation.sort_values("reduction_pp", ascending=False).iloc[0]

    def format_frame(frame: pd.DataFrame, columns: list[str]) -> str:
        return "\n".join(
            f"- {row[columns[0]]}: "
            + ", ".join(
                f"{column}={row[column]:.2%}" if "rate" in column else f"{column}={row[column]:,.0f}"
                for column in columns[1:]
            )
            for _, row in frame.iterrows()
        )

    report = f"""# Delivery Performance Summary

## Beginner

- Delivered on-time rate: {on_time:.2%}
- Average review score for on-time orders: {review_on_time:.2f}
- Average review score for late orders: {review_late:.2f}

Top customer states by on-time rate with at least 500 delivered orders:
{format_frame(top_regions, ["segment_value", "on_time_rate", "delivered_orders"])}

Lowest customer states by on-time rate with at least 500 delivered orders:
{format_frame(bottom_regions, ["segment_value", "on_time_rate", "delivered_orders"])}

## Intermediate

- Reliability score is a weighted composite of on-time rate, review score, cancellation control, seller handoff speed, and delivery volume confidence.

Top ranked sellers with at least 10 delivered orders:
{format_frame(top_sellers, ["seller_id", "reliability_score", "on_time_rate", "delivered_orders"])}

Repeat-purchase rates by first-order delivery experience:
{format_frame(repeat_rates, ["first_order_delivery_segment", "repeat_rate", "customers"])}

Delivery stage benchmark snapshot:
{stage_benchmarks[["delivery_experience", "avg_seller_processing_days", "avg_carrier_delivery_days", "avg_review_score"]].to_string(index=False)}

## Advanced

- Baseline late-delivery risk model ROC AUC: {metrics["roc_auc"]:.3f}
- Baseline late-delivery risk model average precision: {metrics["average_precision"]:.3f}
- High-risk queue precision at the operating threshold: {metrics["high_risk_precision"]:.2%}
- High-risk queue recall at the operating threshold: {metrics["high_risk_recall"]:.2%}
- Best simulated intervention: {best_intervention["scenario"]} with a {best_intervention["reduction_pp"]:.2f} percentage-point late-rate reduction.

These results are prototype-level and meant to guide the next dashboard or notebook iteration, not act as production policy.
"""

    (REPORTS_DIR / "delivery_performance_summary.md").write_text(report)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    items = build_order_items_enriched(
        data["order_items"],
        data["products"],
        data["sellers"],
        data["category_translation"],
    )
    delivery_orders = build_delivery_orders(
        data["orders"],
        data["customers"],
        data["reviews"],
        data["payments"],
        items,
    )
    seller_order_delivery = build_seller_order_delivery(items, delivery_orders)
    seller_reliability = build_seller_reliability(seller_order_delivery)
    customer_retention, retention_summary = build_customer_retention(delivery_orders)
    driver_summary, regional_summary, stage_benchmarks = build_driver_summaries(delivery_orders)
    predictions, model_metrics = train_delivery_risk_model(delivery_orders)
    intervention_simulation = simulate_interventions(predictions)
    seller_recommendations = build_seller_recommendations(seller_reliability)

    outputs = {
        "delivery_order_features.csv": delivery_orders,
        "seller_order_delivery.csv": seller_order_delivery,
        "seller_reliability_rankings.csv": seller_reliability,
        "customer_retention_by_cohort.csv": customer_retention,
        "customer_retention_summary.csv": retention_summary,
        "delivery_driver_summary.csv": driver_summary,
        "regional_delivery_summary.csv": regional_summary,
        "delivery_stage_benchmarks.csv": stage_benchmarks,
        "delivery_risk_predictions.csv": predictions,
        "delivery_model_metrics.csv": model_metrics,
        "delivery_intervention_simulation.csv": intervention_simulation,
        "seller_action_recommendations.csv": seller_recommendations,
    }

    for filename, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / filename, index=False)

    write_summary_report(
        delivery_orders,
        regional_summary,
        stage_benchmarks,
        seller_reliability,
        retention_summary,
        model_metrics,
        intervention_simulation,
    )

    print(f"Saved {len(outputs)} delivery analysis files to {PROCESSED_DIR}")
    print(f"Saved summary report to {REPORTS_DIR / 'delivery_performance_summary.md'}")


if __name__ == "__main__":
    main()
