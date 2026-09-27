# Raw data

The raw Olist CSVs (~240 MB) are not committed to this repository.

## Download

1. [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist)

## Expected layout

```
data/raw/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
├── product_category_name_translation.csv
└── marketing_funnel/
    ├── olist_closed_deals_dataset.csv
    └── olist_marketing_qualified_leads_dataset.csv
```

Then run `python src/build_delivery_performance_assets.py` to regenerate everything in `data/processed/`.
