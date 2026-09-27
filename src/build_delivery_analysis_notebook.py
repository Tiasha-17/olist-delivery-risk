from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / "delivery-performance-analysis.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    source = dedent(text).strip("\n")
    if source:
        source += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """
            # Delivery Performance Analysis

            This notebook is the main analysis deliverable for the delivery-performance project.
            It is organized into three layers:

            - Beginner: How delivery timing affects customer reviews and where performance differs by region.
            - Intermediate: A seller reliability framework with rankings, retention, and operational root causes.
            - Advanced: A prototype delivery-risk system, intervention simulation, and seller action queue.
            """
        ),
        md(
            """
            ## How To Use This Notebook

            1. Run `python3 scripts/build_delivery_performance_assets.py` if the processed files are missing.
            2. Run the cells from top to bottom.
            3. Use the tables and charts here as the basis for slides, dashboards, or a project report.

            The notebook relies on the processed files inside `data/processed/`, not on ad hoc joins.
            """
        ),
        code(
            """
            from pathlib import Path
            import os

            def find_project_root(start: Path) -> Path:
                for candidate in [start.resolve(), *start.resolve().parents]:
                    if (candidate / "data" / "processed").exists():
                        return candidate
                raise FileNotFoundError("Could not find the project root containing data/processed.")

            ROOT = find_project_root(Path.cwd())
            os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp" / "matplotlib")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.ticker import PercentFormatter

            sns.set_theme(style="whitegrid", palette="deep")
            plt.rcParams["figure.figsize"] = (11, 6)
            plt.rcParams["axes.titlesize"] = 14
            plt.rcParams["axes.labelsize"] = 11
            pd.set_option("display.max_columns", 100)
            pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
            """
        ),
        code(
            """
            PROCESSED = ROOT / "data" / "processed"

            delivery_orders = pd.read_csv(
                PROCESSED / "delivery_order_features.csv",
                parse_dates=[
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "min_shipping_limit_date",
                    "max_shipping_limit_date",
                    "purchase_month",
                ],
            )
            seller_rankings = pd.read_csv(PROCESSED / "seller_reliability_rankings.csv")
            seller_order_delivery = pd.read_csv(
                PROCESSED / "seller_order_delivery.csv",
                parse_dates=[
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ],
            )
            retention = pd.read_csv(
                PROCESSED / "customer_retention_by_cohort.csv",
                parse_dates=["cohort_month"],
            )
            retention_summary = pd.read_csv(PROCESSED / "customer_retention_summary.csv")
            regional_summary = pd.read_csv(PROCESSED / "regional_delivery_summary.csv")
            driver_summary = pd.read_csv(PROCESSED / "delivery_driver_summary.csv")
            stage_benchmarks = pd.read_csv(PROCESSED / "delivery_stage_benchmarks.csv")
            predictions = pd.read_csv(
                PROCESSED / "delivery_risk_predictions.csv",
                parse_dates=["order_purchase_timestamp"],
            )
            model_metrics = pd.read_csv(PROCESSED / "delivery_model_metrics.csv")
            interventions = pd.read_csv(PROCESSED / "delivery_intervention_simulation.csv")
            seller_actions = pd.read_csv(PROCESSED / "seller_action_recommendations.csv")

            delivered = delivery_orders[delivery_orders["delivered_flag"] == True].copy()
            eligible_sellers = seller_rankings[seller_rankings["eligible_for_ranking"] == True].copy()

            dataset_overview = pd.DataFrame(
                {
                    "dataset": [
                        "delivery_orders",
                        "delivered_orders",
                        "seller_rankings",
                        "eligible_sellers",
                        "retention_rows",
                        "risk_predictions",
                    ],
                    "rows": [
                        len(delivery_orders),
                        len(delivered),
                        len(seller_rankings),
                        len(eligible_sellers),
                        len(retention),
                        len(predictions),
                    ],
                }
            )
            dataset_overview
            """
        ),
        md(
            """
            ## Executive Snapshot

            Before diving into details, start with the overall business picture:

            - How often orders arrive on time
            - How strongly lateness affects customer reviews
            - Which regions are doing well or poorly
            - Whether the advanced prototype surfaces a usable high-risk queue
            """
        ),
        code(
            """
            executive_kpis = pd.DataFrame(
                {
                    "metric": [
                        "Delivered orders",
                        "On-time delivery rate",
                        "Late delivery rate",
                        "Average review score for on-time orders",
                        "Average review score for late orders",
                        "Eligible sellers in reliability ranking",
                        "Risk-model ROC AUC",
                        "High-risk precision",
                    ],
                    "value": [
                        len(delivered),
                        delivered["on_time_flag"].mean(),
                        delivered["late_flag"].mean(),
                        delivered.loc[delivered["on_time_flag"] == True, "review_score"].mean(),
                        delivered.loc[delivered["late_flag"] == True, "review_score"].mean(),
                        len(eligible_sellers),
                        float(model_metrics.loc[model_metrics["metric"] == "roc_auc", "value"].iloc[0]),
                        float(model_metrics.loc[model_metrics["metric"] == "high_risk_precision", "value"].iloc[0]),
                    ],
                }
            )
            executive_kpis
            """
        ),
        code(
            """
            customer_states = regional_summary.query("segment_type == 'customer_state' and delivered_orders >= 500").copy()
            best_states = customer_states.nlargest(5, "on_time_rate")[
                ["segment_value", "delivered_orders", "on_time_rate", "avg_review_score"]
            ]
            worst_states = customer_states.nsmallest(5, "on_time_rate")[
                ["segment_value", "delivered_orders", "on_time_rate", "avg_review_score"]
            ]

            print("Top customer states by on-time performance")
            print(best_states.to_string(index=False))
            print("\\nLowest customer states by on-time performance")
            print(worst_states.to_string(index=False))
            """
        ),
        md(
            """
            ## Beginner Analysis

            The beginner track answers the core business question:
            **How does delivery performance affect customer reviews, and where do we see the biggest regional differences?**
            """
        ),
        code(
            """
            delay_analysis = delivered.copy()
            delay_analysis["delay_bucket"] = pd.cut(
                delay_analysis["delay_days"],
                bins=[-np.inf, -5, 0, 3, 7, np.inf],
                labels=[
                    "5+ days early",
                    "1-5 days early",
                    "1-3 days late",
                    "4-7 days late",
                    "8+ days late",
                ],
            )

            review_by_delay = (
                delay_analysis.groupby("delay_bucket", observed=False)
                .agg(
                    orders=("order_id", "nunique"),
                    avg_review_score=("review_score", "mean"),
                    avg_delay_days=("delay_days", "mean"),
                )
                .reset_index()
            )
            review_by_delay
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            sns.barplot(
                data=review_by_delay,
                x="delay_bucket",
                y="avg_review_score",
                ax=axes[0],
                color="#3b82f6",
            )
            axes[0].set_title("Average Review Score By Delivery Delay Bucket")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("Average review score")
            axes[0].tick_params(axis="x", rotation=20)

            sns.barplot(
                data=review_by_delay,
                x="delay_bucket",
                y="orders",
                ax=axes[1],
                color="#10b981",
            )
            axes[1].set_title("Delivered Order Volume By Delay Bucket")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("Delivered orders")
            axes[1].tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            monthly_trend = (
                delivered.groupby("purchase_month", as_index=False)
                .agg(
                    delivered_orders=("order_id", "nunique"),
                    on_time_rate=("on_time_flag", "mean"),
                    avg_review_score=("review_score", "mean"),
                )
                .sort_values("purchase_month")
            )

            fig, ax1 = plt.subplots(figsize=(13, 5))
            ax2 = ax1.twinx()

            ax1.plot(monthly_trend["purchase_month"], monthly_trend["on_time_rate"], color="#2563eb", linewidth=2.5)
            ax2.plot(monthly_trend["purchase_month"], monthly_trend["avg_review_score"], color="#f97316", linewidth=2.5)

            ax1.set_title("Monthly Trend: On-Time Delivery vs Average Review Score")
            ax1.set_ylabel("On-time rate")
            ax2.set_ylabel("Average review score")
            ax1.yaxis.set_major_formatter(PercentFormatter(1))
            ax1.set_xlabel("")

            plt.tight_layout()
            plt.show()

            monthly_trend.tail(12)
            """
        ),
        code(
            """
            customer_state_plot = customer_states.copy()
            customer_state_plot["label_group"] = np.where(
                customer_state_plot["on_time_rate"] >= customer_state_plot["on_time_rate"].median(),
                "Higher on-time states",
                "Lower on-time states",
            )

            fig, ax = plt.subplots(figsize=(12, 7))
            sns.scatterplot(
                data=customer_state_plot,
                x="delivered_orders",
                y="on_time_rate",
                hue="label_group",
                size="total_payment_value",
                sizes=(50, 450),
                alpha=0.8,
                ax=ax,
            )

            for _, row in pd.concat(
                [customer_state_plot.nlargest(4, "on_time_rate"), customer_state_plot.nsmallest(4, "on_time_rate")]
            ).drop_duplicates("segment_value").iterrows():
                ax.text(row["delivered_orders"], row["on_time_rate"], row["segment_value"], fontsize=9)

            ax.set_title("Regional Performance: Delivery Volume vs On-Time Rate")
            ax.set_xlabel("Delivered orders")
            ax.set_ylabel("On-time rate")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            route_type = driver_summary.query("segment_type == 'route_type'").copy()
            top_categories = (
                driver_summary.query("segment_type == 'primary_category' and delivered_orders >= 800")
                .sort_values("on_time_rate", ascending=False)
                .head(10)
            )

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            sns.barplot(
                data=route_type,
                x="segment_value",
                y="on_time_rate",
                ax=axes[0],
                color="#6366f1",
            )
            axes[0].set_title("On-Time Rate By Route Type")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("On-time rate")
            axes[0].yaxis.set_major_formatter(PercentFormatter(1))

            sns.barplot(
                data=top_categories.sort_values("on_time_rate"),
                x="on_time_rate",
                y="segment_value",
                ax=axes[1],
                color="#14b8a6",
            )
            axes[1].set_title("Top Categories By On-Time Rate (>= 800 delivered orders)")
            axes[1].set_xlabel("On-time rate")
            axes[1].set_ylabel("")
            axes[1].xaxis.set_major_formatter(PercentFormatter(1))

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ### Beginner Takeaway

            The early story is usually very clear in this dataset:

            - Late deliveries are associated with materially lower review scores.
            - Regional performance is uneven, especially when comparing high-volume states.
            - Some route and product-category combinations are consistently easier to deliver reliably than others.
            """
        ),
        md(
            """
            ## Intermediate Analysis

            The intermediate track builds a framework that can support operational management:

            - Seller reliability ranking
            - Customer retention by first-order delivery experience
            - Root-cause diagnostics for seller delays
            """
        ),
        code(
            """
            seller_tier_distribution = (
                eligible_sellers.groupby("reliability_tier", as_index=False)
                .agg(
                    sellers=("seller_id", "nunique"),
                    avg_on_time_rate=("on_time_rate", "mean"),
                    avg_review_score=("avg_review_score", "mean"),
                    avg_delivered_orders=("delivered_orders", "mean"),
                )
                .sort_values("avg_on_time_rate", ascending=False)
            )

            print("Reliability tier distribution")
            print(seller_tier_distribution.to_string(index=False))

            print("\\nTop 10 eligible sellers")
            print(
                eligible_sellers[
                    [
                        "seller_id",
                        "seller_state",
                        "delivered_orders",
                        "on_time_rate",
                        "avg_review_score",
                        "reliability_score",
                        "dominant_root_cause",
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )

            print("\\nBottom 10 eligible sellers")
            print(
                eligible_sellers[
                    [
                        "seller_id",
                        "seller_state",
                        "delivered_orders",
                        "on_time_rate",
                        "avg_review_score",
                        "reliability_score",
                        "dominant_root_cause",
                    ]
                ]
                .tail(10)
                .to_string(index=False)
            )
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(12, 7))
            scatter = eligible_sellers.copy()
            scatter["bubble_size"] = np.clip(scatter["delivered_orders"], 10, 300)

            sns.scatterplot(
                data=scatter,
                x="delivered_orders",
                y="reliability_score",
                hue="dominant_root_cause",
                size="bubble_size",
                sizes=(30, 350),
                alpha=0.75,
                ax=ax,
            )
            ax.set_title("Seller Reliability Score vs Delivered Volume")
            ax.set_xlabel("Delivered orders")
            ax.set_ylabel("Reliability score")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            root_cause_summary = (
                eligible_sellers.groupby("dominant_root_cause", as_index=False)
                .agg(
                    sellers=("seller_id", "nunique"),
                    avg_reliability_score=("reliability_score", "mean"),
                    avg_on_time_rate=("on_time_rate", "mean"),
                    avg_seller_processing_days=("avg_seller_processing_days", "mean"),
                    avg_carrier_delivery_days=("avg_carrier_delivery_days", "mean"),
                    avg_cancellation_rate=("cancellation_rate", "mean"),
                )
                .sort_values("sellers", ascending=False)
            )
            root_cause_summary
            """
        ),
        code(
            """
            benchmark_plot = stage_benchmarks.copy()
            benchmark_plot = benchmark_plot[benchmark_plot["delivery_experience"] != "unknown"]

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            sns.barplot(
                data=benchmark_plot,
                x="delivery_experience",
                y="avg_seller_processing_days",
                ax=axes[0],
                color="#f59e0b",
            )
            axes[0].set_title("Seller Handoff Time By Delivery Experience")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("Average seller processing days")
            axes[0].tick_params(axis="x", rotation=20)

            sns.barplot(
                data=benchmark_plot,
                x="delivery_experience",
                y="avg_carrier_delivery_days",
                ax=axes[1],
                color="#ef4444",
            )
            axes[1].set_title("Carrier Transit Time By Delivery Experience")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("Average carrier delivery days")
            axes[1].tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            retention_window = retention[retention["months_since_first"] <= 6].copy()
            order = [
                "on_time_or_early",
                "late_1to3_days",
                "late_4plus_days",
                "not_delivered",
                "unknown",
            ]
            retention_pivot = (
                retention_window.pivot_table(
                    index="first_order_delivery_segment",
                    columns="months_since_first",
                    values="retention_rate",
                    aggfunc="mean",
                )
                .reindex(order)
            )

            plt.figure(figsize=(11, 4.5))
            sns.heatmap(retention_pivot, annot=True, fmt=".1%", cmap="YlGnBu")
            plt.title("Customer Retention Heatmap By First-Order Delivery Experience")
            plt.xlabel("Months since first purchase")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()

            retention_summary.sort_values("repeat_rate", ascending=False)
            """
        ),
        code(
            """
            critical_actions = seller_actions[seller_actions["priority"].isin(["critical", "high"])].copy()
            critical_actions.head(15)
            """
        ),
        md(
            """
            ### Intermediate Takeaway

            The reliability score is useful because it combines customer outcome, operational speed, and scale.
            Instead of only ranking sellers by late rate, it highlights **why** performance is weak:

            - slow seller handoff
            - slow carrier transit
            - approval latency
            - cancellations
            """
        ),
        md(
            """
            ## Advanced Analysis

            The advanced track uses the project assets as a prototype operating system:

            - Predictive delivery-risk model
            - Dynamic promise-engine simulation
            - Seller performance monitoring queue
            """
        ),
        code(
            """
            model_metrics
            """
        ),
        code(
            """
            prediction_analysis = predictions.copy()
            prediction_analysis["risk_decile"] = pd.qcut(
                prediction_analysis["predicted_late_risk"].rank(method="first"),
                10,
                labels=[f"D{i}" for i in range(1, 11)],
            )

            decile_summary = (
                prediction_analysis.groupby("risk_decile", observed=False)
                .agg(
                    orders=("order_id", "nunique"),
                    avg_predicted_risk=("predicted_late_risk", "mean"),
                    actual_late_rate=("actual_late_flag", "mean"),
                )
                .reset_index()
            )
            decile_summary
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 5))
            decile_plot = decile_summary.copy()
            x = np.arange(len(decile_plot))

            ax.plot(x, decile_plot["avg_predicted_risk"], marker="o", linewidth=2.5, label="Average predicted risk")
            ax.plot(x, decile_plot["actual_late_rate"], marker="o", linewidth=2.5, label="Actual late rate")

            ax.set_xticks(x)
            ax.set_xticklabels(decile_plot["risk_decile"])
            ax.set_title("Risk Deciles: Predicted Risk vs Actual Late Rate")
            ax.set_xlabel("Risk decile")
            ax.set_ylabel("Rate")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 5))
            intervention_plot = interventions.sort_values("reduction_pp", ascending=False).copy()

            sns.barplot(
                data=intervention_plot,
                x="reduction_pp",
                y="scenario",
                ax=ax,
                color="#8b5cf6",
            )
            ax.set_title("Simulated Late-Rate Reduction By Intervention")
            ax.set_xlabel("Reduction in late rate (percentage points)")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.show()

            intervention_plot
            """
        ),
        code(
            """
            platform_queue = seller_actions[
                [
                    "seller_id",
                    "seller_state",
                    "priority",
                    "reliability_score",
                    "dominant_root_cause",
                    "monitor_metric",
                    "recommended_action",
                ]
            ].head(20)
            platform_queue
            """
        ),
        md(
            """
            ## Final Interpretation

            This notebook gives you a full project path:

            - **Beginner**: show that late delivery hurts reviews and that performance varies meaningfully by state, route type, and category.
            - **Intermediate**: use the reliability score to rank sellers, explain operational bottlenecks, and connect bad delivery experiences to lower repeat behavior.
            - **Advanced**: use the risk model and intervention simulation as the foundation for a delivery-risk system, a dynamic promise engine, and a seller-performance platform.

            If you want to present this work, the strongest slide sequence is:

            1. On-time vs late reviews
            2. Regional performance gaps
            3. Seller reliability tiers and root causes
            4. Retention by first-order delivery experience
            5. Risk-model queue and intervention simulation
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
