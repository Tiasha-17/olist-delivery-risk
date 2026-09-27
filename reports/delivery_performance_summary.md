# Delivery Performance Summary

## Beginner

- Delivered on-time rate: 91.88%
- Average review score for on-time orders: 4.29
- Average review score for late orders: 2.57

Top customer states by on-time rate with at least 500 delivered orders:
- PR: on_time_rate=95.00%, delivered_orders=4,923
- MG: on_time_rate=94.39%, delivered_orders=11,354
- SP: on_time_rate=94.09%, delivered_orders=40,501

Lowest customer states by on-time rate with at least 500 delivered orders:
- BA: on_time_rate=85.96%, delivered_orders=3,256
- CE: on_time_rate=84.68%, delivered_orders=1,279
- MA: on_time_rate=80.33%, delivered_orders=717

## Intermediate

- Reliability score is a weighted composite of on-time rate, review score, cancellation control, seller handoff speed, and delivery volume confidence.

Top ranked sellers with at least 10 delivered orders:
- 725c32fa80c2faacc4fc88450d27314e: reliability_score=96, on_time_rate=98.88%, delivered_orders=89
- 334cab711dee080b079fa5779b584783: reliability_score=96, on_time_rate=100.00%, delivered_orders=11
- d566c37fa119d5e66c4e9052e83ee4ea: reliability_score=96, on_time_rate=96.92%, delivered_orders=65
- 41c2bad7229b0c25e6becf179ebf63ff: reliability_score=95, on_time_rate=100.00%, delivered_orders=20
- acce39e832338debb07b02385cde5967: reliability_score=95, on_time_rate=100.00%, delivered_orders=44

Repeat-purchase rates by first-order delivery experience:
- on_time_or_early: repeat_rate=1.89%, customers=85,659
- late_4plus_days: repeat_rate=1.52%, customers=5,016
- not_delivered: repeat_rate=2.18%, customers=2,840
- late_1to3_days: repeat_rate=1.40%, customers=2,573
- unknown: repeat_rate=0.00%, customers=8

Delivery stage benchmark snapshot:
delivery_experience  avg_seller_processing_days  avg_carrier_delivery_days  avg_review_score
     late_1to3_days                    4.333393                  16.215929          3.766882
    late_4plus_days                    5.829613                  30.563315          1.936816
   on_time_or_early                    2.576766                   7.886819          4.294292
            unknown                    3.177892                        NaN          4.500000

## Advanced

- Baseline late-delivery risk model ROC AUC: 0.740
- Baseline late-delivery risk model average precision: 0.114
- High-risk queue precision at the operating threshold: 15.12%
- High-risk queue recall at the operating threshold: 11.17%
- Best simulated intervention: high_risk_expedite_2_days with a 0.19 percentage-point late-rate reduction.

These results are prototype-level and meant to guide the next dashboard or notebook iteration, not act as production policy.
