# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Sales AI — Gold Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook turns the Silver `sales_integrated` table into a single,
# MAGIC model-ready Gold table for the sales forecasting model.
# MAGIC
# MAGIC Three things happen here, in order, and the order matters:
# MAGIC
# MAGIC 1. **Aggregate to the forecasting grain** — `date + product_id + store_id`.
# MAGIC    Silver is transaction-level; the model needs one row per day per series.
# MAGIC 2. **Fill every missing calendar day** — around 3% of product-store-days
# MAGIC    have no transaction at all (nothing sold that day). If lag/rolling
# MAGIC    features are computed on the existing rows as-is, `lag_1` would skip
# MAGIC    straight past a gap and quietly grab the last day something *did*
# MAGIC    sell — not actually "yesterday". Filling every date with an explicit
# MAGIC    zero-sales row first makes every later lag/rolling calculation
# MAGIC    correct by construction. This step **adds** rows — nothing is dropped.
# MAGIC 3. **Engineer features** on top of that gap-free daily series — lag,
# MAGIC    rolling, trend, price, promotion, store, product, and interaction
# MAGIC    features — using only information available up to and including
# MAGIC    each row's own date (no future leakage).
# MAGIC
# MAGIC This notebook builds a **single global feature table** — every category
# MAGIC and store together, with `category` / `product_id` / `store_id` included
# MAGIC as model features — because we're training one global model rather than
# MAGIC one model per category.
# MAGIC
# MAGIC Output: `sales_ai.gold.sales_features`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports and Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "sales_ai"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper — Row Count Audit
# MAGIC
# MAGIC Same helper used in the Silver notebook, redefined here so this notebook
# MAGIC stays self-contained. Note the message it prints changes meaning in this
# MAGIC notebook: in Silver, "removed" rows meant a data-quality problem; here,
# MAGIC a **negative** removed count (i.e. rows increased) is expected and
# MAGIC correct, because the scaffold step deliberately adds missing-day rows.

# COMMAND ----------

def audit_step(df_before, df_after, step_name):
    """
    Print the row count before and after a transformation step, and the
    change in row count. Returns df_after unchanged so this can be used
    inline without breaking a transformation chain.
    """
    before_count = df_before.count()
    after_count = df_after.count()
    change = after_count - before_count

    print(f"[{step_name}] before={before_count:,}  after={after_count:,}  change={change:+,}")
    return df_after

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Silver Data

# COMMAND ----------

silver_df = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.sales_integrated")
products_df = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.products")
stores_df = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.stores")
calendar_df = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.calendar_marketing_external")

print(f"Silver integrated records : {silver_df.count():,}")
print(f"Products                  : {products_df.count()}")
print(f"Stores                    : {stores_df.count()}")
print(f"Calendar days             : {calendar_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Aggregate to the Forecasting Grain
# MAGIC
# MAGIC `channel_id` is normalized defensively here (Title Case) before the
# MAGIC channel-mix aggregation below, in case any capitalization inconsistency
# MAGIC from the source data made it through — this keeps "Online" and "ONLINE"
# MAGIC from being counted as two different channels.
# MAGIC
# MAGIC Measures are summed to the daily grain (so this also correctly absorbs
# MAGIC the rare exact-duplicate transaction rows noted in the source data
# MAGIC quality report — summing two identical duplicate rows for the same key
# MAGIC would double-count, so duplicates are collapsed with an inner
# MAGIC deduplication on the full row content before aggregating).

# COMMAND ----------

display(silver_df)

# COMMAND ----------

silver_df = silver_df.withColumn("channel_id", F.initcap(F.trim(F.col("channel_id"))))

# Collapse exact-content duplicate rows (same transaction, different transaction_id)
# before aggregating, so they don't get double-counted by the sum() below.
dedup_cols = [c for c in silver_df.columns if c not in ("transaction_id", "silver_processed_timestamp")]
silver_df_dedup = silver_df.dropDuplicates(dedup_cols)
silver_df_dedup = audit_step(silver_df, silver_df_dedup, "collapse exact-content duplicate rows")

daily_agg = (
    silver_df_dedup
    .groupBy("date", "product_id", "store_id")
    .agg(
        F.sum("units_sold").alias("units_sold"),
        F.sum("gross_sales").alias("gross_sales"),
        F.sum("discount_amount").alias("discount_amount"),
        F.sum("net_sales").alias("net_sales"),
        F.sum("return_units").alias("return_units"),
        F.sum("return_amount").alias("return_amount"),
        F.avg("unit_price").alias("unit_price"),
        F.avg("discount_percentage").alias("discount_percentage"),
        F.avg("inventory_available").alias("inventory_available"),
        F.max(F.col("stockout_flag").cast("int")).cast("boolean").alias("stockout_flag"),
        F.first("promotion_id", ignorenulls=True).alias("promotion_id"),
        # Channel mix — how the day's units broke down across channels
        F.sum(F.when(F.col("channel_id") == "Online", F.col("units_sold")).otherwise(0)).alias("online_units"),
        F.sum(F.when(F.col("channel_id") == "Physical Store", F.col("units_sold")).otherwise(0)).alias("physical_units"),
        F.sum(F.when(F.col("channel_id") == "Mobile App", F.col("units_sold")).otherwise(0)).alias("mobile_units"),
        F.sum(F.when(F.col("channel_id") == "Marketplace", F.col("units_sold")).otherwise(0)).alias("marketplace_units"),
    )
)

print(f"Daily grain records (date + product_id + store_id): {daily_agg.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build a Complete Daily Calendar Scaffold
# MAGIC
# MAGIC For every `(product_id, store_id)` pair that actually appears in the
# MAGIC data, this builds one row for **every date** from that series' true
# MAGIC start (the later of the product's launch date and the store's opening
# MAGIC date) through the last date in the calendar — regardless of whether a
# MAGIC sale happened that day. `daily_agg` is then left-joined onto this
# MAGIC scaffold, and missing-day measures are filled in as explicit zeros
# MAGIC (a day with no transaction is a real zero-sales day, not a missing
# MAGIC value).
# MAGIC
# MAGIC The combos are derived from what's actually observed in the data
# MAGIC (rather than assumed from a full product × store cross-join), so this
# MAGIC works correctly even if the product assortment isn't fully dense.

# COMMAND ----------

# Observed product-store combinations, plus each series' true start date
observed_combos = daily_agg.select("product_id", "store_id").distinct()

combo_bounds = (
    observed_combos
    .join(products_df.select("product_id", "launch_date"), "product_id", "left")
    .join(stores_df.select("store_id", "opening_date"), "store_id", "left")
    .withColumn("series_start_date", F.greatest("launch_date", "opening_date"))
    .select("product_id", "store_id", "series_start_date")
)

calendar_max_date = calendar_df.agg(F.max("date")).first()[0]

# Cross join each combo with every calendar date, then trim to that combo's valid range
scaffold = (
    combo_bounds
    .crossJoin(calendar_df.select("date"))
    .filter(F.col("date") >= F.col("series_start_date"))
    .filter(F.col("date") <= F.lit(calendar_max_date))
    .select("product_id", "store_id", "date")
)

print(f"Scaffold records (every calendar day per active series): {scaffold.count():,}")

# COMMAND ----------

display(scaffold)

# COMMAND ----------

gold_daily = (
    scaffold
    .join(daily_agg, ["date", "product_id", "store_id"], "left")
    # Zero-fill genuine zero-sales days -- a missing row here means "nothing sold", not "unknown"
    .fillna({
        "units_sold": 0, "gross_sales": 0, "discount_amount": 0, "net_sales": 0,
        "return_units": 0, "return_amount": 0, "online_units": 0, "physical_units": 0,
        "mobile_units": 0, "marketplace_units": 0, "discount_percentage": 0.0,
    })
    .withColumn("stockout_flag", F.coalesce(F.col("stockout_flag"), F.lit(False)))
)

gold_daily = audit_step(daily_agg, gold_daily, "fill missing calendar days (rows should increase)")

# COMMAND ----------

display(gold_daily)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Join Dimension Attributes
# MAGIC
# MAGIC The scaffold-filled zero-sales rows still need their product, store,
# MAGIC and calendar context (category, region, holiday flag, etc.) — that
# MAGIC context doesn't come from the fact table, so it's joined back in here.

# COMMAND ----------

product_columns = [
    "product_id", "category", "subcategory", "brand", "lifecycle_status",
    "launch_date", "base_price", "cost_price", "demand_class", "price_elasticity_tier",
]

store_columns = [
    "store_id", "store_type", "region", "city", "city_tier",
    "store_size_sqft", "customer_density", "average_income_index",
]

calendar_columns = [
    "date", "day_of_week", "day_name", "week_of_year", "month", "quarter", "year",
    "is_weekend", "holiday_flag", "holiday_type", "major_event_flag",
    "temperature_c", "rainfall_mm", "inflation_index", "consumer_confidence_index",
    "campaign_active_flag",
]

gold_df = (
    gold_daily
    .join(products_df.select(product_columns), "product_id", "left")
    .join(stores_df.select(store_columns), "store_id", "left")
    .join(calendar_df.select(calendar_columns), "date", "left")
)

print(f"Records after dimension join: {gold_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Time-Series Feature Engineering
# MAGIC
# MAGIC From here on, every feature is computed with a `Window` ordered by
# MAGIC `date` and partitioned by `(product_id, store_id)` — i.e. one
# MAGIC independent time line per series. Because the scaffold made every
# MAGIC series date-complete, a row-based window here is equivalent to a true
# MAGIC time-based window: `lag_1` really is yesterday.
# MAGIC
# MAGIC Rolling windows use `rowsBetween(-N, -1)` — up to but **not including**
# MAGIC the current row — so no feature ever uses the value it's trying to help
# MAGIC predict.

# COMMAND ----------

series_window = Window.partitionBy("product_id", "store_id").orderBy("date")

# --- Lag features -----------------------------------------------------
gold_df = (
    gold_df
    .withColumn("lag_1", F.lag("units_sold", 1).over(series_window))
    .withColumn("lag_7", F.lag("units_sold", 7).over(series_window))
    .withColumn("lag_14", F.lag("units_sold", 14).over(series_window))
    .withColumn("lag_28", F.lag("units_sold", 28).over(series_window))
    .withColumn("lag_365", F.lag("units_sold", 365).over(series_window))
)

print("Lag features added: lag_1, lag_7, lag_14, lag_28, lag_365")

# COMMAND ----------

# MAGIC %md
# MAGIC `lag_365` is null for every series' first year by definition — there's
# MAGIC no day 365 days earlier to look back to yet. Those nulls are **left in
# MAGIC place** rather than dropped or filled with 0 (a 0 would falsely imply
# MAGIC "no sales a year ago" and would materially bias the model). Gradient
# MAGIC boosting frameworks like LightGBM and XGBoost handle nulls natively by
# MAGIC learning the best split direction for missing values, so this is safe
# MAGIC to leave as-is — see the null audit in Section 8.

# COMMAND ----------

# --- Rolling window features -------------------------------------------
rolling_7 = series_window.rowsBetween(-7, -1)
rolling_14 = series_window.rowsBetween(-14, -1)
rolling_28 = series_window.rowsBetween(-28, -1)
rolling_90 = series_window.rowsBetween(-90, -1)

gold_df = (
    gold_df
    .withColumn("rolling_mean_7", F.avg("units_sold").over(rolling_7))
    .withColumn("rolling_mean_14", F.avg("units_sold").over(rolling_14))
    .withColumn("rolling_mean_28", F.avg("units_sold").over(rolling_28))
    .withColumn("rolling_mean_90", F.avg("units_sold").over(rolling_90))
    .withColumn("rolling_std_7", F.stddev("units_sold").over(rolling_7))
    .withColumn("rolling_std_28", F.stddev("units_sold").over(rolling_28))
)

print("Rolling features added: rolling_mean_7/14/28/90, rolling_std_7/28")

# COMMAND ----------

# --- Trend features -----------------------------------------------------
# Compares this window's average to the equivalent window a period earlier,
# so a positive value means demand is accelerating, not just "high".
gold_df = (
    gold_df
    .withColumn("trend_7d", F.col("rolling_mean_7") - F.lag("rolling_mean_7", 7).over(series_window))
    .withColumn("trend_28d", F.col("rolling_mean_28") - F.lag("rolling_mean_28", 28).over(series_window))
    .withColumn(
        "yoy_growth_pct",
        F.when(
            F.col("lag_365") > 0,
            F.round((F.col("units_sold") - F.col("lag_365")) / F.col("lag_365") * 100, 2)
        )
    )
    .withColumn(
        "mom_growth_pct",
        F.when(
            F.lag("rolling_mean_28", 28).over(series_window) > 0,
            F.round(
                (F.col("rolling_mean_28") - F.lag("rolling_mean_28", 28).over(series_window))
                / F.lag("rolling_mean_28", 28).over(series_window) * 100,
                2
            )
        )
    )
)

print("Trend features added: trend_7d, trend_28d, yoy_growth_pct, mom_growth_pct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Price Features
# MAGIC
# MAGIC `unit_price` is null on zero-sales days (no transaction means no
# MAGIC observed price that day). A forward-filled `last_known_unit_price` is
# MAGIC computed first — using only past and current values, so it carries no
# MAGIC leakage — and all price-change features are built from that instead of
# MAGIC the raw (gappy) `unit_price` column.

# COMMAND ----------

forward_fill_window = series_window.rowsBetween(Window.unboundedPreceding, 0)

gold_df = gold_df.withColumn(
    "last_known_unit_price",
    F.last("unit_price", ignorenulls=True).over(forward_fill_window)
)

gold_df = (
    gold_df
    .withColumn("price_change", F.round(
        F.col("last_known_unit_price") - F.lag("last_known_unit_price", 1).over(series_window), 2
    ))
    .withColumn(
        "price_change_pct",
        F.when(
            F.lag("last_known_unit_price", 1).over(series_window) > 0,
            F.round(F.col("price_change") / F.lag("last_known_unit_price", 1).over(series_window) * 100, 2)
        )
    )
)

# Same-product price relative to its own average price across all stores that same day.
# (This dataset uses one flagship product per category, so "category average" and
# "product average across stores" are the same thing here -- kept as category-partitioned
# for compatibility with a future catalog that has multiple products per category.)
category_price_window = Window.partitionBy("category", "date")
gold_df = gold_df.withColumn(
    "price_vs_category_avg",
    F.round(F.col("last_known_unit_price") / F.avg("last_known_unit_price").over(category_price_window), 3)
)

print("Price features added: last_known_unit_price, price_change, price_change_pct, price_vs_category_avg")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Promotion Features
# MAGIC
# MAGIC `days_since_last_promotion` and `promotion_frequency_90d` are derived
# MAGIC from this series' own transaction history (past only — no leakage).
# MAGIC `days_since/until_campaign` come from the calendar's known campaign
# MAGIC schedule instead: real retailers plan promotional calendars in advance,
# MAGIC so "a sale event starts in 5 days" is a legitimate, already-known input
# MAGIC at prediction time — not a forecast target leaking into a feature.

# COMMAND ----------

gold_df = gold_df.withColumn("promotion_flag", F.col("promotion_id").isNotNull().cast("integer"))

# Days since this series' own last promoted sale (past-only, per product-store)
promo_date_col = F.when(F.col("promotion_flag") == 1, F.col("date"))
last_promo_window = series_window.rowsBetween(Window.unboundedPreceding, -1)

gold_df = (
    gold_df
    .withColumn("_last_promo_date", F.last(promo_date_col, ignorenulls=True).over(last_promo_window))
    .withColumn("days_since_last_promotion", F.datediff("date", "_last_promo_date"))
    .withColumn("promotion_frequency_90d", F.sum("promotion_flag").over(series_window.rowsBetween(-90, -1)))
    .drop("_last_promo_date")
)

# Calendar-level known campaign schedule (category-agnostic, computed once over the small calendar table)
calendar_order = Window.orderBy("date")
campaign_date_col = F.when(F.col("campaign_active_flag"), F.col("date"))

calendar_campaign_features = (
    calendar_df
    .select("date", "campaign_active_flag")
    .withColumn("_campaign_date", campaign_date_col)
    .withColumn(
        "days_since_last_campaign",
        F.datediff("date", F.last("_campaign_date", ignorenulls=True).over(calendar_order.rowsBetween(Window.unboundedPreceding, -1)))
    )
    .withColumn(
        "days_until_next_campaign",
        F.datediff(
            F.first("_campaign_date", ignorenulls=True).over(calendar_order.rowsBetween(1, Window.unboundedFollowing)),
            "date"
        )
    )
    .select("date", "days_since_last_campaign", "days_until_next_campaign")
)

gold_df = gold_df.join(calendar_campaign_features, "date", "left")

print("Promotion features added: promotion_flag, days_since_last_promotion, "
      "promotion_frequency_90d, days_since_last_campaign, days_until_next_campaign")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Store, Product, and Interaction Features

# COMMAND ----------

# Store x category historical average -- expanding (past-only) average so early rows
# aren't influenced by sales that haven't happened yet.
store_category_window = (
    Window.partitionBy("store_id", "category")
    .orderBy("date")
    .rowsBetween(Window.unboundedPreceding, -1)
)

gold_df = gold_df.withColumn(
    "store_category_historical_avg_units",
    F.round(F.avg("units_sold").over(store_category_window), 2)
)

# Product age in days as of this row's date
gold_df = gold_df.withColumn("product_age_days", F.datediff("date", "launch_date"))

# A couple of explicit interaction terms. Gradient boosting trees can learn most
# interactions automatically from the raw features, so this list is intentionally
# short -- illustrative rather than exhaustive.
gold_df = gold_df.withColumn(
    "promo_weekend_interaction",
    F.col("promotion_flag") * F.col("is_weekend").cast("integer")
)

print("Store, product, and interaction features added.")

# COMMAND ----------

display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Train / Validation / Test Split
# MAGIC
# MAGIC A time-based split column, not a random one — this is a forecasting
# MAGIC problem, so validation and test periods must come strictly after the
# MAGIC training period.

# COMMAND ----------

gold_df = gold_df.withColumn(
    "dataset_split",
    F.when(F.col("year") <= 2023, "train")
    .when(F.col("year") == 2024, "validation")
    .otherwise("test")
)

gold_df.groupBy("dataset_split").count().orderBy(
    F.when(F.col("dataset_split") == "train", 1)
    .when(F.col("dataset_split") == "validation", 2)
    .otherwise(3)
).show()

# COMMAND ----------

display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Null Audit
# MAGIC
# MAGIC Expected nulls, by design:
# MAGIC - `lag_365` / `yoy_growth_pct` — null for each series' first 365 days
# MAGIC - `lag_1/7/14/28` and rolling features — null for the first few days of each series
# MAGIC - `price_change*` — null for each series' very first row (no prior price to compare to)
# MAGIC
# MAGIC These are left as nulls rather than filled — see Section 4 for why.
# MAGIC A large null count *outside* the first year of a series would indicate
# MAGIC a real bug and is worth checking for here before this table is trusted
# MAGIC downstream.

# COMMAND ----------

null_audit_columns = [
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_365",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
    "price_change", "price_change_pct", "yoy_growth_pct", "mom_growth_pct",
]

null_counts = gold_df.select(
    [F.sum(F.col(c).isNull().cast("integer")).alias(c) for c in null_audit_columns]
)
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Channel Mix Shares
# MAGIC
# MAGIC Computed last since they depend on the final `units_sold` column.
# MAGIC Left null (not 0) on true zero-sales days, since a channel share isn't
# MAGIC meaningful when nothing sold anywhere that day.

# COMMAND ----------

gold_df = (
    gold_df
    .withColumn("online_share", F.when(F.col("units_sold") > 0, F.round(F.col("online_units") / F.col("units_sold"), 3)))
    .withColumn("physical_share", F.when(F.col("units_sold") > 0, F.round(F.col("physical_units") / F.col("units_sold"), 3)))
    .withColumn("mobile_share", F.when(F.col("units_sold") > 0, F.round(F.col("mobile_units") / F.col("units_sold"), 3)))
    .withColumn("marketplace_share", F.when(F.col("units_sold") > 0, F.round(F.col("marketplace_units") / F.col("units_sold"), 3)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Final Column Selection

# COMMAND ----------

gold_final_columns = [
    # Grain / keys
    "date", "product_id", "store_id", "dataset_split",

    # Target
    "units_sold",

    # Raw daily measures
    "gross_sales", "discount_amount", "net_sales", "return_units", "return_amount",
    "last_known_unit_price", "discount_percentage", "inventory_available", "stockout_flag",

    # Channel mix
    "online_units", "physical_units", "mobile_units", "marketplace_units",
    "online_share", "physical_share", "mobile_share", "marketplace_share",

    # Product features
    "category", "subcategory", "brand", "lifecycle_status", "demand_class",
    "price_elasticity_tier", "base_price", "cost_price", "product_age_days",

    # Store features
    "store_type", "region", "city", "city_tier", "store_size_sqft",
    "customer_density", "average_income_index",

    # Calendar features
    "day_of_week", "day_name", "week_of_year", "month", "quarter", "year", "is_weekend",
    "holiday_flag", "holiday_type", "major_event_flag",
    "temperature_c", "rainfall_mm", "inflation_index", "consumer_confidence_index",

    # Lag features
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_365",

    # Rolling features
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_mean_90",
    "rolling_std_7", "rolling_std_28",

    # Trend features
    "trend_7d", "trend_28d", "yoy_growth_pct", "mom_growth_pct",

    # Price features
    "price_change", "price_change_pct", "price_vs_category_avg",

    # Promotion features
    "promotion_flag", "days_since_last_promotion", "promotion_frequency_90d",
    "days_since_last_campaign", "days_until_next_campaign",

    # Store performance feature
    "store_category_historical_avg_units",

    # Interaction feature
    "promo_weekend_interaction",
]

gold_df = gold_df.select(gold_final_columns)

print(f"Final Gold column count: {len(gold_df.columns)}")
print(f"Final Gold row count   : {gold_df.count():,}")

# COMMAND ----------

display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validate Grain Uniqueness
# MAGIC
# MAGIC The single most important check for a forecasting feature table: there
# MAGIC must be exactly one row per `(date, product_id, store_id)`. If this
# MAGIC returns anything other than 0, a join earlier in this notebook produced
# MAGIC unwanted fan-out and must be fixed before this table is used for
# MAGIC training.

# COMMAND ----------

duplicate_grain_count = (
    gold_df.groupBy("date", "product_id", "store_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print(f"Duplicate (date, product_id, store_id) keys: {duplicate_grain_count} (expected: 0)")
assert duplicate_grain_count == 0, "Grain is not unique -- investigate before writing to Gold."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Write Gold Table

# COMMAND ----------

(
    gold_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("dataset_split")
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.sales_features")
)

print("Gold Delta table created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Gold Table

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN sales_ai.gold;

# COMMAND ----------

display(
    spark.table("sales_ai.gold.sales_features")
    .orderBy("product_id", "store_id", "date")
    .limit(20)
)

# COMMAND ----------

print("Gold transformation pipeline completed successfully.")
print(f"sales_ai.gold.sales_features is ready for model training "
      f"({len(gold_final_columns)} columns, {gold_df.count():,} rows).")