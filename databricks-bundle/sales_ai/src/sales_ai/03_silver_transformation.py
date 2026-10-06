# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Sales AI — Silver Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook cleans, validates, standardizes, and integrates the Bronze datasets into a trusted Silver analytical dataset.
# MAGIC
# MAGIC Source layers:
# MAGIC - sales_transactions
# MAGIC - products
# MAGIC - stores
# MAGIC - calendar_marketing_external
# MAGIC
# MAGIC Output:
# MAGIC - Cleaned Silver dimension tables
# MAGIC - Integrated Silver sales dataset
# MAGIC
# MAGIC Every cleaning step below prints a before/after record count, so if a run
# MAGIC unexpectedly drops a large number of rows, it shows up immediately in the
# MAGIC notebook output instead of silently disappearing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports and Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "sales_ai"

BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper — Row Count Audit
# MAGIC
# MAGIC Small helper used throughout this notebook to print how many rows a
# MAGIC filtering step removed. This keeps every cleaning step auditable without
# MAGIC repeating the same three lines everywhere.

# COMMAND ----------

def audit_step(df_before, df_after, step_name):
    """
    Print the row count before and after a cleaning step, along with how
    many rows were removed. Returns df_after unchanged so this can be used
    inline without breaking a transformation chain.
    """
    before_count = df_before.count()
    after_count = df_after.count()
    removed = before_count - after_count

    print(f"[{step_name}] before={before_count:,}  after={after_count:,}  removed={removed:,}")
    return df_after

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Bronze Tables

# COMMAND ----------

sales_df = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.sales_transactions")
products_df = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.products")
stores_df = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.stores")
calendar_df = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.calendar_marketing_external")

print(f"Bronze sales records    : {sales_df.count():,}")
print(f"Bronze product records  : {products_df.count():,}")
print(f"Bronze store records    : {stores_df.count():,}")
print(f"Bronze calendar records : {calendar_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardize Data Types
# MAGIC
# MAGIC Bronze columns come in with Spark's inferred schema, which is not always
# MAGIC reliable for numeric and boolean columns loaded from CSV. This step casts
# MAGIC every column to its intended type before any business-rule filtering
# MAGIC happens, so downstream comparisons (e.g. `units_sold >= 0`) behave
# MAGIC predictably.

# COMMAND ----------

# Sales — cast measures to double, flags to boolean, date to date
sales_df = (
    sales_df
    .withColumn("date", F.to_date("date"))
    .withColumn("units_sold", F.col("units_sold").cast("double"))
    .withColumn("unit_price", F.col("unit_price").cast("double"))
    .withColumn("gross_sales", F.col("gross_sales").cast("double"))
    .withColumn("discount_amount", F.col("discount_amount").cast("double"))
    .withColumn("net_sales", F.col("net_sales").cast("double"))
    .withColumn("discount_percentage", F.col("discount_percentage").cast("double"))
    .withColumn("inventory_available", F.col("inventory_available").cast("double"))
    .withColumn("return_units", F.col("return_units").cast("double"))
    .withColumn("return_amount", F.col("return_amount").cast("double"))
    .withColumn("true_demand_estimate", F.col("true_demand_estimate").cast("double"))
    .withColumn("stockout_flag", F.col("stockout_flag").cast("boolean"))
)

# Products — cast price and lifecycle metrics to double, launch_date to date
products_df = (
    products_df
    .withColumn("launch_date", F.to_date("launch_date"))
    .withColumn("base_price", F.col("base_price").cast("double"))
    .withColumn("cost_price", F.col("cost_price").cast("double"))
    .withColumn("return_rate_baseline", F.col("return_rate_baseline").cast("double"))
    .withColumn("trend_annual_multiplier", F.col("trend_annual_multiplier").cast("double"))
)

# Stores — cast geographic and sizing attributes
stores_df = (
    stores_df
    .withColumn("opening_date", F.to_date("opening_date"))
    .withColumn("store_size_sqft", F.col("store_size_sqft").cast("double"))
    .withColumn("latitude", F.col("latitude").cast("double"))
    .withColumn("longitude", F.col("longitude").cast("double"))
    .withColumn("customer_density", F.col("customer_density").cast("double"))
    .withColumn("average_income_index", F.col("average_income_index").cast("double"))
    .withColumn("city_tier", F.col("city_tier").cast("integer"))
)

# Calendar — cast weather, economic and marketing measures, and boolean flags
calendar_df = (
    calendar_df
    .withColumn("date", F.to_date("date"))
    .withColumn("temperature_c", F.col("temperature_c").cast("double"))
    .withColumn("rainfall_mm", F.col("rainfall_mm").cast("double"))
    .withColumn("humidity_pct", F.col("humidity_pct").cast("double"))
    .withColumn("inflation_index", F.col("inflation_index").cast("double"))
    .withColumn("consumer_confidence_index", F.col("consumer_confidence_index").cast("double"))
    .withColumn("unemployment_rate", F.col("unemployment_rate").cast("double"))
    .withColumn("fuel_price_index", F.col("fuel_price_index").cast("double"))
    .withColumn("digital_ad_spend", F.col("digital_ad_spend").cast("double"))
    .withColumn("offline_ad_spend", F.col("offline_ad_spend").cast("double"))
    .withColumn("holiday_flag", F.col("holiday_flag").cast("boolean"))
    .withColumn("major_event_flag", F.col("major_event_flag").cast("boolean"))
    .withColumn("campaign_active_flag", F.col("campaign_active_flag").cast("boolean"))
)

print("Data types standardized.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deduplication
# MAGIC
# MAGIC Duplicate business keys are removed here. `dropDuplicates` keeps one
# MAGIC arbitrary row per key — acceptable for this dataset since duplicates are
# MAGIC exact content copies (see the data quality report from the source
# MAGIC dataset), but worth revisiting with a `ROW_NUMBER()` tie-breaker
# MAGIC (e.g. latest ingestion timestamp) if a future source ever produces
# MAGIC duplicates that genuinely differ in content.

# COMMAND ----------

_sales_before = sales_df
sales_df = sales_df.dropDuplicates(["transaction_id"])
sales_df = audit_step(_sales_before, sales_df, "sales: drop duplicate transaction_id")

_products_before = products_df
products_df = products_df.dropDuplicates(["product_id"])
products_df = audit_step(_products_before, products_df, "products: drop duplicate product_id")

_stores_before = stores_df
stores_df = stores_df.dropDuplicates(["store_id"])
stores_df = audit_step(_stores_before, stores_df, "stores: drop duplicate store_id")

_calendar_before = calendar_df
calendar_df = calendar_df.dropDuplicates(["date"])
calendar_df = audit_step(_calendar_before, calendar_df, "calendar: drop duplicate date")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Critical Fields
# MAGIC
# MAGIC Rows missing a value in any column required to identify or join the
# MAGIC record are dropped. Missing values in *non-critical* columns (e.g.
# MAGIC `promotion_id`, `inventory_available`) are left as nulls — they carry
# MAGIC business meaning ("no promotion active") and should not be removed.

# COMMAND ----------

_sales_before = sales_df
sales_df = sales_df.filter(
    F.col("transaction_id").isNotNull()
    & F.col("date").isNotNull()
    & F.col("product_id").isNotNull()
    & F.col("store_id").isNotNull()
    & F.col("units_sold").isNotNull()
    & F.col("net_sales").isNotNull()
)
sales_df = audit_step(_sales_before, sales_df, "sales: drop rows missing critical fields")

_products_before = products_df
products_df = products_df.filter(
    F.col("product_id").isNotNull()
    & F.col("product_name").isNotNull()
    & F.col("category").isNotNull()
    & F.col("base_price").isNotNull()
)
products_df = audit_step(_products_before, products_df, "products: drop rows missing critical fields")

_stores_before = stores_df
stores_df = stores_df.filter(
    F.col("store_id").isNotNull()
    & F.col("store_name").isNotNull()
    & F.col("city").isNotNull()
    & F.col("region").isNotNull()
)
stores_df = audit_step(_stores_before, stores_df, "stores: drop rows missing critical fields")

_calendar_before = calendar_df
calendar_df = calendar_df.filter(F.col("date").isNotNull())
calendar_df = audit_step(_calendar_before, calendar_df, "calendar: drop rows missing date")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Business Rule Validation
# MAGIC
# MAGIC Numeric business values are validated against simple, non-negotiable
# MAGIC rules (e.g. sales figures cannot be negative, discount percentage must
# MAGIC fall between 0 and 100). Rows that fail are dropped rather than
# MAGIC corrected, since a negative sale or an out-of-range discount indicates
# MAGIC a source data problem rather than something safe to guess a fix for.

# COMMAND ----------

_sales_before = sales_df
sales_df = sales_df.filter(
    (F.col("units_sold") >= 0)
    & (F.col("unit_price") >= 0)
    & (F.col("gross_sales") >= 0)
    & (F.col("discount_amount") >= 0)
    & (F.col("net_sales") >= 0)
    & F.col("discount_percentage").between(0, 100)
    & (F.col("return_units") >= 0)
    & (F.col("return_amount") >= 0)
)
sales_df = audit_step(_sales_before, sales_df, "sales: drop rows failing business rules")

_products_before = products_df
products_df = products_df.filter(
    (F.col("base_price") >= 0)
    & (F.col("cost_price") >= 0)
)
products_df = audit_step(_products_before, products_df, "products: drop rows failing business rules")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Referential Integrity
# MAGIC
# MAGIC Sales rows are inner-joined against the cleaned product and store
# MAGIC dimensions so that no sales record can reference a `product_id` or
# MAGIC `store_id` that doesn't exist in Silver. Any row dropped here points to
# MAGIC a genuine upstream data problem and is worth investigating if the
# MAGIC removed count is ever more than a handful of rows.

# COMMAND ----------

valid_products = products_df.select("product_id").distinct()
valid_stores = stores_df.select("store_id").distinct()

_sales_before = sales_df
sales_df = (
    sales_df
    .join(valid_products, "product_id", "inner")
    .join(valid_stores, "store_id", "inner")
)
sales_df = audit_step(_sales_before, sales_df, "sales: drop rows failing referential integrity")

print("\nData quality checks completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Integrate Sales, Product, Store and Calendar Data
# MAGIC
# MAGIC The cleaned fact table is left-joined against the three cleaned
# MAGIC dimensions to produce a single, wide, analysis-ready table. Only the
# MAGIC columns actually needed downstream (EDA, feature engineering) are
# MAGIC selected from each dimension, to keep the integrated table's schema
# MAGIC intentional rather than accidental.

# COMMAND ----------

product_columns = [
    "product_id",
    "product_name",
    "category",
    "subcategory",
    "brand",
    "product_type",
    "launch_date",
    "lifecycle_status",
    "base_price",
    "cost_price",
    "seasonality_type",
    "demand_class",
    "return_rate_baseline",
    "trend_annual_multiplier",
    "price_elasticity_tier",
    "category_seasonality_flavor"
]

store_columns = [
    "store_id",
    "store_name",
    "city",
    "state",
    "region",
    "country",
    "store_type",
    "store_size_sqft",
    "opening_date",
    "sales_channel",
    "customer_density",
    "average_income_index",
    "urban_rural",
    "store_performance_segment",
    "city_tier"
]

calendar_columns = [
    "date",
    "day_of_week",
    "day_name",
    "week_of_year",
    "month",
    "month_name",
    "quarter",
    "year",
    "day_of_month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "holiday_name",
    "holiday_flag",
    "holiday_type",
    "days_to_holiday",
    "days_from_holiday",
    "promotion_event",
    "major_event_flag",
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "weather_condition",
    "inflation_index",
    "consumer_confidence_index",
    "unemployment_rate",
    "fuel_price_index",
    "marketing_campaign",
    "campaign_type",
    "campaign_intensity",
    "digital_ad_spend",
    "offline_ad_spend",
    "campaign_active_flag"
]

silver_sales_df = (
    sales_df
    .join(products_df.select(product_columns), "product_id", "left")
    .join(stores_df.select(store_columns), "store_id", "left")
    .join(calendar_df.select(calendar_columns), "date", "left")
)

print(f"Integrated records: {silver_sales_df.count():,}")

# COMMAND ----------

display(silver_sales_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Business Metrics
# MAGIC
# MAGIC Derived metrics that don't exist in any single source table, but are
# MAGIC cheap to compute once here rather than recomputing in every downstream
# MAGIC notebook.

# COMMAND ----------

silver_sales_df = (

    silver_sales_df

    # Gross margin = net sales minus the cost of goods sold, rounded to 2 decimal places
    .withColumn(
        "gross_margin",
        F.round(F.col("net_sales") - (F.col("units_sold") * F.col("cost_price")), 2)
    )

    # Return rate = returned units as a share of units sold (0 when nothing sold)
    .withColumn(
        "return_rate",
        F.round(
            F.when(
                F.col("units_sold") > 0,
                F.col("return_units") / F.col("units_sold")
            ).otherwise(0.0),
            2
        )
    )

    # Realized unit price = actual average selling price after discounts
    .withColumn(
        "realized_unit_price",
        F.round(
            F.when(
                F.col("units_sold") > 0,
                F.col("net_sales") / F.col("units_sold")
            ).otherwise(0.0),
            2
        )
    )

    # Total marketing spend = digital + offline, treating missing spend as zero
    .withColumn(
        "total_ad_spend",
        F.round(
            F.coalesce(F.col("digital_ad_spend"), F.lit(0.0))
            + F.coalesce(F.col("offline_ad_spend"), F.lit(0.0)),
            2
        )
    )

    # Audit column: when this Silver row was produced
    .withColumn("silver_processed_timestamp", F.current_timestamp())
)

# COMMAND ----------

display(silver_sales_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Integrated Dataset
# MAGIC
# MAGIC A final summary check before writing to Delta. `unique_transactions`
# MAGIC should equal `total_records` (transaction_id is the grain of this
# MAGIC table); if it doesn't, the join above introduced fan-out and needs
# MAGIC investigating before this table is trusted downstream.

# COMMAND ----------

quality_summary = silver_sales_df.select(
    F.count("*").alias("total_records"),
    F.countDistinct("transaction_id").alias("unique_transactions"),
    F.countDistinct("product_id").alias("unique_products"),
    F.countDistinct("store_id").alias("unique_stores"),
    F.min("date").alias("min_date"),
    F.max("date").alias("max_date"),
    F.sum(F.when(F.col("net_sales").isNull(), 1).otherwise(0)).alias("null_net_sales")
)

display(quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Silver Delta Tables

# COMMAND ----------

# Cleaned dimension tables
(
    products_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.products")
)

(
    stores_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.stores")
)

(
    calendar_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.calendar_marketing_external")
)

# Integrated analytical dataset (fact + dimensions, one row per transaction)
(
    silver_sales_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.sales_integrated")
)

print("Silver Delta tables created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Silver Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN sales_ai.silver;

# COMMAND ----------

display(
    spark.table("sales_ai.silver.sales_integrated").limit(20)
)

# COMMAND ----------

print("Silver transformation pipeline completed successfully.")

# COMMAND ----------

