from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MARKETING_RAW_DIR = RAW_DIR / "marketing_funnel"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "mql": pd.read_csv(
            MARKETING_RAW_DIR / "olist_marketing_qualified_leads_dataset.csv",
            parse_dates=["first_contact_date"],
        ),
        "closed_deals": pd.read_csv(
            MARKETING_RAW_DIR / "olist_closed_deals_dataset.csv",
            parse_dates=["won_date"],
        ),
        "sellers": pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv"),
        "order_items": pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv"),
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
        "reviews": pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv"),
    }


def build_marketing_funnel(mql: pd.DataFrame, closed_deals: pd.DataFrame) -> pd.DataFrame:
    marketing_funnel = mql.merge(
        closed_deals,
        on="mql_id",
        how="left",
        validate="one_to_one",
    )
    marketing_funnel["is_closed_deal"] = marketing_funnel["seller_id"].notna()
    marketing_funnel["lead_to_win_days"] = (
        marketing_funnel["won_date"] - marketing_funnel["first_contact_date"]
    ).dt.days
    return marketing_funnel


def build_seller_performance(
    sellers: pd.DataFrame,
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    reviews_by_order = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "nunique"),
        )
    )

    seller_orders = (
        order_items.groupby(["seller_id", "order_id"], as_index=False)
        .agg(
            items_in_order=("order_item_id", "count"),
            order_revenue=("price", "sum"),
            order_freight=("freight_value", "sum"),
            unique_products_in_order=("product_id", "nunique"),
        )
        .merge(
            orders[
                [
                    "order_id",
                    "order_status",
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                ]
            ],
            on="order_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            reviews_by_order,
            on="order_id",
            how="left",
            validate="many_to_one",
        )
    )

    seller_metrics = (
        seller_orders.assign(
            delivered_flag=seller_orders["order_status"].eq("delivered").astype(int),
            canceled_flag=seller_orders["order_status"].eq("canceled").astype(int),
        )
        .groupby("seller_id", as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),
            total_items_sold=("items_in_order", "sum"),
            total_revenue=("order_revenue", "sum"),
            total_freight=("order_freight", "sum"),
            avg_order_value=("order_revenue", "mean"),
            avg_items_per_order=("items_in_order", "mean"),
            avg_review_score=("review_score", "mean"),
            delivered_orders=("delivered_flag", "sum"),
            canceled_orders=("canceled_flag", "sum"),
            first_order_at=("order_purchase_timestamp", "min"),
            last_order_at=("order_purchase_timestamp", "max"),
        )
    )

    active_days = (
        seller_metrics["last_order_at"] - seller_metrics["first_order_at"]
    ).dt.days.fillna(0)
    seller_metrics["active_days"] = active_days
    seller_metrics["active_months"] = (active_days / 30).clip(lower=1).round(2)
    seller_metrics["avg_monthly_revenue"] = (
        seller_metrics["total_revenue"] / seller_metrics["active_months"]
    ).round(2)

    seller_performance = sellers.merge(
        seller_metrics,
        on="seller_id",
        how="left",
        validate="one_to_one",
    )
    seller_performance["has_order_history"] = seller_performance["total_orders"].fillna(0).gt(0)
    return seller_performance


def build_marketing_sellers_enriched(
    marketing_funnel: pd.DataFrame,
    seller_performance: pd.DataFrame,
) -> pd.DataFrame:
    marketing_sellers = marketing_funnel.loc[
        marketing_funnel["is_closed_deal"]
    ].copy()
    marketing_sellers = marketing_sellers.merge(
        seller_performance,
        on="seller_id",
        how="left",
        validate="one_to_one",
    )
    marketing_sellers["is_active_in_ecommerce_dataset"] = marketing_sellers[
        "has_order_history"
    ].eq(True)
    return marketing_sellers


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)


def main() -> None:
    inputs = load_inputs()

    marketing_funnel = build_marketing_funnel(
        inputs["mql"],
        inputs["closed_deals"],
    )
    seller_performance = build_seller_performance(
        inputs["sellers"],
        inputs["order_items"],
        inputs["orders"],
        inputs["reviews"],
    )
    marketing_sellers_enriched = build_marketing_sellers_enriched(
        marketing_funnel,
        seller_performance,
    )

    write_outputs(
        {
            "marketing_funnel": marketing_funnel,
            "seller_performance": seller_performance,
            "marketing_sellers_enriched": marketing_sellers_enriched,
        }
    )

    print(f"Saved marketing_funnel: {len(marketing_funnel):,} rows")
    print(f"Saved seller_performance: {len(seller_performance):,} rows")
    print(f"Saved marketing_sellers_enriched: {len(marketing_sellers_enriched):,} rows")
    print(
        "Marketing sellers with order history:",
        int(marketing_sellers_enriched["is_active_in_ecommerce_dataset"].sum()),
    )


if __name__ == "__main__":
    main()
