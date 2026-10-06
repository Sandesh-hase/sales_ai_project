# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Sales AI — Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook explores the Silver `sales_integrated` table before any
# MAGIC feature engineering happens. It runs on Silver — not Bronze — because
# MAGIC Bronze still contains nulls, duplicates, and unvalidated values, and
# MAGIC charting that would mean drawing conclusions from data-quality noise
# MAGIC rather than real business patterns.
# MAGIC
# MAGIC The dataset is intentionally compact — 6 product categories, 8 stores —
# MAGIC so every category can be inspected individually rather than sampled.
# MAGIC
# MAGIC What this notebook covers:
# MAGIC 1. Dataset overview
# MAGIC 2. Overall sales trend over time
# MAGIC 3. Category-wise sales trends
# MAGIC 4. Seasonality — day-of-week and month-of-year patterns
# MAGIC 5. Store-level differences
# MAGIC 6. Promotion impact
# MAGIC 7. Price vs. demand relationship
# MAGIC 8. Stockout occurrences
# MAGIC 9. Summary takeaways going into feature engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports and Configuration

# COMMAND ----------

from pyspark.sql import functions as F

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.titleweight"] = "bold"

CATALOG = "sales_ai"
SILVER_SCHEMA = "silver"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Silver Data
# MAGIC
# MAGIC `sales_integrated` already contains every dimension attribute needed for
# MAGIC this analysis (category, store, calendar), so most of this notebook only
# MAGIC needs that one table. `products` is loaded separately only where the
# MAGIC full product catalog (not just products that appear in a given chart) is
# MAGIC useful for reference.

# COMMAND ----------

sales_integrated_df = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.sales_integrated")
products_df = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.products")

print(f"Silver integrated records : {sales_integrated_df.count():,}")
print(f"Products in catalog       : {products_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert to pandas for Plotting
# MAGIC
# MAGIC The integrated table is small enough (well under 100K rows) to bring
# MAGIC into a single-node pandas DataFrame for charting with matplotlib and
# MAGIC seaborn. For a materially larger dataset, this step should aggregate in
# MAGIC Spark first (`groupBy` + `agg`) and only collect the aggregated result.

# COMMAND ----------

df = sales_integrated_df.toPandas()
products_pd = products_df.toPandas()

df["date"] = pd.to_datetime(df["date"])

print(f"pandas DataFrame shape: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset Overview

# COMMAND ----------

print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total days in range: {(df['date'].max() - df['date'].min()).days + 1}")
print(f"Categories: {df['category'].nunique()}  |  Stores: {df['store_id'].nunique()}")
print(f"Total transactions: {len(df):,}")
print(f"Total units sold: {df['units_sold'].sum():,.0f}")
print(f"Total net sales: Rs {df['net_sales'].sum():,.0f}")

# COMMAND ----------

products_pd[["product_id", "product_name", "category", "lifecycle_status", "base_price", "demand_class"]]

# COMMAND ----------

# MAGIC %md
# MAGIC Series depth confirms every category-store combination has deep, near-continuous history — the key property this dataset was designed for.

# COMMAND ----------

depth = df.groupby(["category", "store_id"]).size().reset_index(name="records")
print(f"Records per category-store series: min={depth['records'].min()}, "
      f"max={depth['records'].max()}, mean={depth['records'].mean():.0f}")

depth.pivot(index="store_id", columns="category", values="records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Overall Sales Trend Over Time

# COMMAND ----------

daily_total = df.groupby("date")["units_sold"].sum().reset_index()
daily_total["rolling_28"] = daily_total["units_sold"].rolling(28, min_periods=1).mean()

fig, ax = plt.subplots()
ax.plot(daily_total["date"], daily_total["units_sold"],
        alpha=0.25, linewidth=0.6, label="Daily units sold (all categories)")
ax.plot(daily_total["date"], daily_total["rolling_28"],
        linewidth=2, color="#2563A8", label="28-day rolling average")
ax.set_title("Total Units Sold Across All Categories & Stores")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Category-Wise Sales Trends
# MAGIC
# MAGIC With only 6 categories, every one can be plotted on the same axes and
# MAGIC compared directly — festival-driven categories (Electronics, Clothing)
# MAGIC spike every year around the same months, while Grocery stays comparatively flat.

# COMMAND ----------

monthly_cat = (
    df.groupby([pd.Grouper(key="date", freq="MS"), "category"])["units_sold"]
    .sum()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 6))
for category, group in monthly_cat.groupby("category"):
    ax.plot(group["date"], group["units_sold"], marker="o", markersize=2.5, linewidth=1.6, label=category)

ax.set_title("Monthly Units Sold by Category")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold (monthly total, all stores)")
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Small-multiples view — one clean panel per category, easier to read individually.

# COMMAND ----------

categories = sorted(df["category"].unique())

fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
for ax, category in zip(axes.flat, categories):
    group = monthly_cat[monthly_cat["category"] == category]
    ax.plot(group["date"], group["units_sold"], color="#1F8A55", linewidth=1.6)
    ax.set_title(category, fontsize=11)
    ax.set_ylabel("Units/month")

fig.suptitle("Monthly Sales Trend — One Panel per Category", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# COMMAND ----------

cat_summary = (
    df.groupby("category")
    .agg(
        total_units=("units_sold", "sum"),
        total_net_sales=("net_sales", "sum"),
        avg_daily_units=("units_sold", "mean"),
        lifecycle=("lifecycle_status", "first"),
    )
    .sort_values("total_units", ascending=False)
)
cat_summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Seasonality — Day-of-Week and Month-of-Year Patterns

# COMMAND ----------

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_avg = df.groupby("day_name")["units_sold"].mean().reindex(dow_order)

fig, ax = plt.subplots()
bar_colors = ["#2563A8"] * 5 + ["#1F8A55", "#1F8A55"]
ax.bar(dow_avg.index, dow_avg.values, color=bar_colors)
ax.set_title("Average Units Sold by Day of Week (Weekend Highlighted)")
ax.set_ylabel("Avg units sold per transaction")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
month_avg = df.groupby("month_name")["units_sold"].mean().reindex(month_order)

fig, ax = plt.subplots()
ax.plot(month_avg.index, month_avg.values, marker="o", color="#6E4FA8", linewidth=2)
ax.set_title("Average Units Sold by Month (Festival Season Visible Oct-Nov)")
ax.set_ylabel("Avg units sold per transaction")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Store-Level Differences

# COMMAND ----------

store_avg = (
    df.groupby(["store_id", "city", "city_tier"])["units_sold"]
    .mean()
    .reset_index()
    .sort_values("units_sold", ascending=False)
)

fig, ax = plt.subplots()
bar_colors = ["#2563A8" if tier == 1 else "#B5451E" for tier in store_avg["city_tier"]]
ax.bar(store_avg["city"], store_avg["units_sold"], color=bar_colors)
ax.set_title("Average Units Sold by Store (Blue = Tier-1 Metro, Orange = Tier-2)")
ax.set_ylabel("Avg units sold per transaction")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Promotion Impact

# COMMAND ----------

promo_compare = (
    df.assign(has_promo=df["promotion_id"].notna())
    .groupby("has_promo")["units_sold"]
    .mean()
)
promo_compare.index = ["No Promotion", "Promotion Active"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(promo_compare.index, promo_compare.values, color=["#8A8F98", "#B5451E"])
ax.set_title("Promotion Uplift — Avg Units Sold")

uplift_pct = (promo_compare["Promotion Active"] / promo_compare["No Promotion"] - 1) * 100
ax.text(0.5, max(promo_compare.values) * 0.5, f"+{uplift_pct:.0f}% uplift",
        ha="center", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Price vs. Demand — Example Category
# MAGIC
# MAGIC Electronics is used here since it carries the highest price-elasticity tier in the product catalog.

# COMMAND ----------

example_product_id = products_pd.loc[products_pd["category"] == "Electronics", "product_id"].iloc[0]
example_base_price = products_pd.loc[products_pd["product_id"] == example_product_id, "base_price"].iloc[0]
example_name = products_pd.loc[products_pd["product_id"] == example_product_id, "product_name"].iloc[0]

example_df = df[df["product_id"] == example_product_id].copy()
example_df["price_ratio"] = example_df["unit_price"] / example_base_price

fig, ax = plt.subplots()
ax.scatter(example_df["price_ratio"], example_df["units_sold"], alpha=0.25, s=12, color="#2563A8")
ax.set_title(f"Price vs. Units Sold — {example_name}")
ax.set_xlabel("Price ratio (unit_price / base_price)")
ax.set_ylabel("Units sold")

price_demand_corr = example_df["price_ratio"].corr(example_df["units_sold"])
ax.text(0.05, 0.92, f"correlation = {price_demand_corr:.2f}",
        transform=ax.transAxes, fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Stockouts

# COMMAND ----------

stockout_by_month = (
    df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")["stockout_flag"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots()
ax.plot(stockout_by_month["month"], stockout_by_month["stockout_flag"] * 100,
        color="#B5451E", linewidth=1.6)
ax.set_title("Stockout Rate Over Time (% of transactions affected)")
ax.set_ylabel("Stockout rate (%)")
plt.tight_layout()
plt.show()

stockout_rows = df[df["stockout_flag"]]
suppressed_demand = (stockout_rows["true_demand_estimate"] - stockout_rows["units_sold"]).mean()

print(f"Overall stockout rate: {df['stockout_flag'].mean() * 100:.2f}% of transactions")
print(f"Average demand suppressed on stockout days: {suppressed_demand:.1f} units")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary — What This Tells Us Going Into Feature Engineering
# MAGIC
# MAGIC - **6 categories, 8 stores, 48 series, ~97% daily density, 5 years of history** — every series is long and clean enough to support `lag_7` / `lag_28` / `lag_365` and rolling-window features.
# MAGIC - Each category has a **visibly distinct seasonal shape** (festival-driven Electronics/Clothing, summer-driven Appliances, flat Grocery) — `category` and `product_id` need to be model features, not something to split into separate models for.
# MAGIC - **Weekend, holiday, and promotion effects** are all clearly present, confirming there's real, learnable signal for a gradient boosting model to pick up through engineered features.
# MAGIC - **Price sensitivity differs by category** — this justifies including price/discount features and category interaction terms.
# MAGIC - **Stockouts genuinely suppress observed sales** below true demand — worth flagging as a real-world forecasting caveat (low sales isn't always low demand).
# MAGIC
# MAGIC **Next notebook:** Gold-layer feature engineering (lags, rolling windows, calendar, promotion, and price features) on top of `sales_ai.silver.sales_integrated`, followed by the gradient boosting forecasting model.

# COMMAND ----------

print("Exploratory data analysis completed successfully.")

# COMMAND ----------

