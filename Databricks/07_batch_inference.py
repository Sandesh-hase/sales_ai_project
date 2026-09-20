# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Sales AI — Batch Inference (30-Day Forecast Generation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook loads the **Champion** model from the Unity Catalog Model
# MAGIC Registry and generates a 30-day-ahead forecast for every
# MAGIC `(product_id, store_id)` series, writing the result to
# MAGIC `sales_ai.gold.forecast_results` — the table the GenAI microservice and
# MAGIC the UI dashboard both read from.
# MAGIC
# MAGIC **This uses recursive (iterative) forecasting**, as discussed earlier:
# MAGIC day+1 is predicted from real history, then that prediction is fed back
# MAGIC in as `lag_1` to predict day+2, and so on out to day+30. This is simple
# MAGIC and effective for a 30-day horizon, but errors can compound the further
# MAGIC out the horizon goes — a **direct multi-step model** (trained to predict
# MAGIC "N days ahead" directly, with `horizon` as a feature) is the natural
# MAGIC production upgrade if 30-day accuracy at the far end of the window ever
# MAGIC becomes a concern.
# MAGIC
# MAGIC A second important simplification: this dataset's calendar
# MAGIC (`calendar_marketing_external`) ends on 2025-12-31, so there is no real
# MAGIC future weather, economic, or planned-campaign data to forecast against.
# MAGIC This notebook assumes a **baseline "no planned promotion" scenario** —
# MAGIC prices held at their last known value, no new promotions, weather and
# MAGIC economic indicators held at their last observed reading. This is a
# MAGIC reasonable default forecast, and it's also exactly the kind of
# MAGIC assumption a "what-if" feature in the GenAI layer could later let a
# MAGIC business user override (e.g. "what would the forecast look like if we
# MAGIC ran a 20% promotion in week 2?").

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports and Configuration

# COMMAND ----------

!pip install lightgbm

# COMMAND ----------

from datetime import timedelta

import numpy as np
import pandas as pd
import mlflow

CATALOG = "sales_ai"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

MODEL_NAME = f"{CATALOG}.{GOLD_SCHEMA}.sales_forecasting_lgbm"
MODEL_ALIAS = "Champion"
FORECAST_HORIZON_DAYS = 30

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the Champion Model
# MAGIC
# MAGIC Loaded via the native `mlflow.lightgbm` flavor rather than the generic
# MAGIC `pyfunc` flavor, so prediction goes straight to the underlying LightGBM
# MAGIC model without passing through pyfunc's input-schema enforcement layer —
# MAGIC this sidesteps the same signature quirk that affected the Unity
# MAGIC Catalog SQL function for this model.

# COMMAND ----------

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.lightgbm.load_model(model_uri)

client = mlflow.MlflowClient()
champion_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
print(f"Loaded model: {MODEL_NAME}, version {champion_version.version} (alias: {MODEL_ALIAS})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Definition
# MAGIC
# MAGIC Must exactly match the feature list and categorical handling used in
# MAGIC `06_model_training.py` (pandas `category` dtype) — a mismatch here
# MAGIC would cause silently wrong predictions rather than an obvious error.

# COMMAND ----------

CATEGORICAL_FEATURES = [
    "product_id", "store_id", "category", "subcategory", "brand",
    "lifecycle_status", "demand_class", "price_elasticity_tier",
    "store_type", "region", "city", "day_name", "holiday_type",
]

NUMERIC_FEATURES = [
    "last_known_unit_price", "discount_percentage", "inventory_available",
    "online_share", "physical_share", "mobile_share", "marketplace_share",
    "base_price", "cost_price", "product_age_days",
    "city_tier", "store_size_sqft", "customer_density", "average_income_index",
    "day_of_week", "week_of_year", "month", "quarter", "year",
    "temperature_c", "rainfall_mm", "inflation_index", "consumer_confidence_index",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_365",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_mean_90",
    "rolling_std_7", "rolling_std_28",
    "trend_7d", "trend_28d", "yoy_growth_pct", "mom_growth_pct",
    "price_change", "price_change_pct", "price_vs_category_avg",
    "promotion_flag", "days_since_last_promotion", "promotion_frequency_90d",
    "days_since_last_campaign", "days_until_next_campaign",
    "store_category_historical_avg_units", "promo_weekend_interaction",
]

BOOLEAN_FEATURES = ["is_weekend", "holiday_flag", "major_event_flag", "stockout_flag"]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Historical Data
# MAGIC
# MAGIC Only the raw (non-engineered) daily measures are needed from history —
# MAGIC lag/rolling/trend features are recomputed fresh at every step of the
# MAGIC recursive loop below, since the loop needs to compute them for
# MAGIC predicted future rows too, not just historical ones. 400 days of
# MAGIC history per series is pulled (more than the 365 needed for `lag_365`,
# MAGIC with a small buffer).

# COMMAND ----------

gold_df = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.sales_features")
max_date = gold_df.agg({"date": "max"}).first()[0]
history_start = max_date - timedelta(days=400)

RAW_HISTORY_COLUMNS = [
    "date", "product_id", "store_id", "units_sold",
    "last_known_unit_price", "discount_percentage", "promotion_flag",
    "inventory_available", "online_share", "physical_share",
    "mobile_share", "marketplace_share", "stockout_flag",
]

history_pd = (
    gold_df
    .filter(gold_df["date"] >= history_start)
    .select(RAW_HISTORY_COLUMNS)
    .toPandas()
)
history_pd["date"] = pd.to_datetime(history_pd["date"])

anchor_date = history_pd["date"].max()
forecast_dates = [anchor_date + timedelta(days=i) for i in range(1, FORECAST_HORIZON_DAYS + 1)]

print(f"History loaded: {len(history_pd):,} rows, {history_pd['date'].min().date()} to {anchor_date.date()}")
print(f"Forecasting {FORECAST_HORIZON_DAYS} days: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Static Product / Store Attributes
# MAGIC
# MAGIC These don't change day-to-day, so they're joined once here rather than
# MAGIC being recomputed inside the forecast loop.

# COMMAND ----------

products_pd = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.products").toPandas()
stores_pd = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.stores").toPandas()

combos = history_pd[["product_id", "store_id"]].drop_duplicates().reset_index(drop=True)

combo_static = (
    combos
    .merge(products_pd[["product_id", "category", "subcategory", "brand", "lifecycle_status",
                         "demand_class", "price_elasticity_tier", "base_price", "cost_price", "launch_date"]],
           on="product_id", how="left")
    .merge(stores_pd[["store_id", "store_type", "region", "city", "city_tier",
                       "store_size_sqft", "customer_density", "average_income_index"]],
           on="store_id", how="left")
)
combo_static["launch_date"] = pd.to_datetime(combo_static["launch_date"])

print(f"Forecasting for {len(combo_static)} product-store series.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Future Calendar Stub
# MAGIC
# MAGIC Day-of-week, month, and similar calendar fields are computed directly
# MAGIC from the date — no external data needed. Holiday flags are set from a
# MAGIC small hardcoded list of fixed-date national holidays that fall inside
# MAGIC this 30-day window (movable festivals aren't included here since they
# MAGIC require a lookahead this project doesn't have data for). Weather and
# MAGIC economic indicators are held at their last observed value, since no
# MAGIC forecast for them exists in this dataset.

# COMMAND ----------

FIXED_FUTURE_HOLIDAYS = {
    # Extend this dict as the forecast horizon moves into new years.
    "2026-01-01": ("New Year", "National"),
    "2026-01-26": ("Republic Day", "National"),
}

last_calendar_row = (
    spark.table(f"{CATALOG}.{GOLD_SCHEMA}.sales_features")
    .filter(f"date = '{anchor_date.date()}'")
    .select("temperature_c", "rainfall_mm", "inflation_index", "consumer_confidence_index")
    .limit(1)
    .toPandas()
    .iloc[0]
)

future_calendar_rows = []
for d in forecast_dates:
    holiday_name, holiday_type = FIXED_FUTURE_HOLIDAYS.get(d.strftime("%Y-%m-%d"), (None, None))
    future_calendar_rows.append({
        "date": d,
        "day_of_week": d.isoweekday(),
        "day_name": d.strftime("%A"),
        "week_of_year": int(d.strftime("%V")),
        "month": d.month,
        "quarter": (d.month - 1) // 3 + 1,
        "year": d.year,
        "is_weekend": d.isoweekday() >= 6,
        "holiday_flag": holiday_name is not None,
        "holiday_type": holiday_type,
        "major_event_flag": False,
        "temperature_c": last_calendar_row["temperature_c"],
        "rainfall_mm": last_calendar_row["rainfall_mm"],
        "inflation_index": last_calendar_row["inflation_index"],
        "consumer_confidence_index": last_calendar_row["consumer_confidence_index"],
    })

future_calendar = pd.DataFrame(future_calendar_rows)
display(future_calendar)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recursive Forecast Loop
# MAGIC
# MAGIC For each future date, every series' feature row is assembled from
# MAGIC `history_pd` (which contains real actuals at the start, and grows by
# MAGIC one predicted row per series per iteration). This is written as a
# MAGIC straightforward nested loop (30 dates x 48 series = 1,440 iterations)
# MAGIC for clarity — at a much larger catalog size, the per-date inner loop
# MAGIC should be vectorized across series the same way the original data
# MAGIC simulation was, rather than iterating row by row.

# COMMAND ----------

def get_lag(hist, product_id, store_id, target_date, lag_days):
    """Look up units_sold exactly `lag_days` before target_date for one series."""
    lookup_date = target_date - timedelta(days=lag_days)
    match = hist[(hist["product_id"] == product_id) & (hist["store_id"] == store_id) & (hist["date"] == lookup_date)]
    return match["units_sold"].iloc[0] if len(match) else np.nan

def get_rolling(hist, product_id, store_id, target_date, window_days, stat="mean"):
    """Rolling mean/std over the window_days strictly before target_date (no leakage)."""
    window_start = target_date - timedelta(days=window_days)
    mask = (
        (hist["product_id"] == product_id) & (hist["store_id"] == store_id)
        & (hist["date"] >= window_start) & (hist["date"] < target_date)
    )
    values = hist.loc[mask, "units_sold"]
    if len(values) == 0:
        return np.nan
    return values.mean() if stat == "mean" else values.std()

def get_last_promo_date(hist, product_id, store_id, before_date):
    mask = (
        (hist["product_id"] == product_id) & (hist["store_id"] == store_id)
        & (hist["date"] < before_date) & (hist["promotion_flag"] == 1)
    )
    promo_dates = hist.loc[mask, "date"]
    return promo_dates.max() if len(promo_dates) else pd.NaT

def get_expanding_avg(hist, store_id, category_product_ids, before_date):
    """Store x category historical average, expanding window up to (not including) before_date."""
    mask = (
        (hist["store_id"] == store_id) & (hist["product_id"].isin(category_product_ids))
        & (hist["date"] < before_date)
    )
    values = hist.loc[mask, "units_sold"]
    return values.mean() if len(values) else np.nan

product_category_map = combo_static.groupby("category")["product_id"].unique().to_dict()

forecast_rows = []
history_running = history_pd.copy()

for horizon_step, target_date in enumerate(forecast_dates, start=1):
    cal_row = future_calendar[future_calendar["date"] == target_date].iloc[0]
    batch_features = []

    for _, combo in combo_static.iterrows():
        pid, sid = combo["product_id"], combo["store_id"]

        last_row = history_running[
            (history_running["product_id"] == pid) & (history_running["store_id"] == sid)
        ].sort_values("date").iloc[-1]

        rolling_mean_7 = get_rolling(history_running, pid, sid, target_date, 7, "mean")
        rolling_mean_28 = get_rolling(history_running, pid, sid, target_date, 28, "mean")
        rolling_mean_7_prior = get_rolling(history_running, pid, sid, target_date - timedelta(days=7), 7, "mean")
        rolling_mean_28_prior = get_rolling(history_running, pid, sid, target_date - timedelta(days=28), 28, "mean")
        lag_365 = get_lag(history_running, pid, sid, target_date, 365)
        units_lag_1 = get_lag(history_running, pid, sid, target_date, 1)

        last_promo_date = get_last_promo_date(history_running, pid, sid, target_date)
        days_since_last_promotion = (target_date - last_promo_date).days if pd.notna(last_promo_date) else np.nan

        promo_window_start = target_date - timedelta(days=90)
        promo_mask = (
            (history_running["product_id"] == pid) & (history_running["store_id"] == sid)
            & (history_running["date"] >= promo_window_start) & (history_running["date"] < target_date)
        )
        promotion_frequency_90d = history_running.loc[promo_mask, "promotion_flag"].sum()

        category_product_ids = product_category_map.get(combo["category"], [pid])
        store_category_hist_avg = get_expanding_avg(history_running, sid, category_product_ids, target_date)

        yoy_growth_pct = (
            (units_lag_1 - lag_365) / lag_365 * 100 if pd.notna(lag_365) and lag_365 > 0 else np.nan
        )
        mom_growth_pct = (
            (rolling_mean_28 - rolling_mean_28_prior) / rolling_mean_28_prior * 100
            if pd.notna(rolling_mean_28_prior) and rolling_mean_28_prior > 0 else np.nan
        )

        # Baseline "no planned promotion" scenario -- see notebook objective above.
        last_known_unit_price = last_row["last_known_unit_price"]

        row = {
            "product_id": pid, "store_id": sid,
            "category": combo["category"], "subcategory": combo["subcategory"], "brand": combo["brand"],
            "lifecycle_status": combo["lifecycle_status"], "demand_class": combo["demand_class"],
            "price_elasticity_tier": combo["price_elasticity_tier"],
            "store_type": combo["store_type"], "region": combo["region"], "city": combo["city"],
            "day_name": cal_row["day_name"], "holiday_type": cal_row["holiday_type"],

            "last_known_unit_price": last_known_unit_price,
            "discount_percentage": 0.0,
            "inventory_available": last_row["inventory_available"],
            "online_share": last_row["online_share"], "physical_share": last_row["physical_share"],
            "mobile_share": last_row["mobile_share"], "marketplace_share": last_row["marketplace_share"],
            "base_price": combo["base_price"], "cost_price": combo["cost_price"],
            "product_age_days": (target_date - combo["launch_date"]).days,
            "city_tier": combo["city_tier"], "store_size_sqft": combo["store_size_sqft"],
            "customer_density": combo["customer_density"], "average_income_index": combo["average_income_index"],
            "day_of_week": cal_row["day_of_week"], "week_of_year": cal_row["week_of_year"],
            "month": cal_row["month"], "quarter": cal_row["quarter"], "year": cal_row["year"],
            "temperature_c": cal_row["temperature_c"], "rainfall_mm": cal_row["rainfall_mm"],
            "inflation_index": cal_row["inflation_index"],
            "consumer_confidence_index": cal_row["consumer_confidence_index"],

            "lag_1": units_lag_1,
            "lag_7": get_lag(history_running, pid, sid, target_date, 7),
            "lag_14": get_lag(history_running, pid, sid, target_date, 14),
            "lag_28": get_lag(history_running, pid, sid, target_date, 28),
            "lag_365": lag_365,

            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_14": get_rolling(history_running, pid, sid, target_date, 14, "mean"),
            "rolling_mean_28": rolling_mean_28,
            "rolling_mean_90": get_rolling(history_running, pid, sid, target_date, 90, "mean"),
            "rolling_std_7": get_rolling(history_running, pid, sid, target_date, 7, "std"),
            "rolling_std_28": get_rolling(history_running, pid, sid, target_date, 28, "std"),

            "trend_7d": rolling_mean_7 - rolling_mean_7_prior if pd.notna(rolling_mean_7_prior) else np.nan,
            "trend_28d": rolling_mean_28 - rolling_mean_28_prior if pd.notna(rolling_mean_28_prior) else np.nan,
            "yoy_growth_pct": yoy_growth_pct,
            "mom_growth_pct": mom_growth_pct,

            "price_change": 0.0,
            "price_change_pct": 0.0,
            "price_vs_category_avg": 1.0,  # no assumed price deviation from the category norm

            "promotion_flag": 0,
            "days_since_last_promotion": days_since_last_promotion,
            "promotion_frequency_90d": promotion_frequency_90d,
            "days_since_last_campaign": np.nan,
            "days_until_next_campaign": np.nan,

            "store_category_historical_avg_units": store_category_hist_avg,
            "promo_weekend_interaction": 0,

            "is_weekend": cal_row["is_weekend"],
            "holiday_flag": cal_row["holiday_flag"],
            "major_event_flag": cal_row["major_event_flag"],
            "stockout_flag": False,
        }
        batch_features.append(row)

    batch_df = pd.DataFrame(batch_features)

    # Cast categorical columns on a SEPARATE copy used only for model.predict().
    # batch_df itself keeps plain string dtype for product_id/store_id/etc., since
    # those columns are reused below to build forecast_id (string concatenation)
    # and to merge with combo_static/stores_pd -- pandas cannot concatenate a
    # `category` dtype column with a string, which is what caused the
    # "unsupported operand type(s) for +: 'Categorical' and 'str'" error.
    predict_df = batch_df.copy()
    for col in CATEGORICAL_FEATURES:
        predict_df[col] = predict_df[col].astype("category")

    predicted_units = np.clip(model.predict(predict_df[FEATURE_COLUMNS]), 0, None)

    batch_df["predicted_units"] = np.round(predicted_units, 2)
    batch_df["forecast_date"] = target_date
    batch_df["forecast_horizon"] = horizon_step

    forecast_rows.append(batch_df[[
        "forecast_date", "product_id", "store_id", "predicted_units", "forecast_horizon"
    ]].copy())

    # Feed this step's predictions back into history so the next iteration's
    # lag/rolling features see them -- this is what makes it "recursive".
    new_history_rows = batch_df[[
        "product_id", "store_id", "last_known_unit_price", "discount_percentage",
        "inventory_available", "online_share", "physical_share", "mobile_share", "marketplace_share",
    ]].copy()
    new_history_rows["date"] = target_date
    new_history_rows["units_sold"] = batch_df["predicted_units"]
    new_history_rows["promotion_flag"] = 0
    new_history_rows["stockout_flag"] = False

    history_running = pd.concat([history_running, new_history_rows[RAW_HISTORY_COLUMNS]], ignore_index=True)

print(f"Recursive forecast complete: {FORECAST_HORIZON_DAYS} days x {len(combo_static)} series "
      f"= {sum(len(r) for r in forecast_rows):,} forecast rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assemble and Write Forecast Results

# COMMAND ----------

forecast_df = pd.concat(forecast_rows, ignore_index=True)
forecast_df["forecast_id"] = (
    forecast_df["product_id"] + "_" + forecast_df["store_id"] + "_"
    + forecast_df["forecast_date"].dt.strftime("%Y%m%d")
)
forecast_df["prediction_timestamp"] = pd.Timestamp.now()
forecast_df["model_name"] = MODEL_NAME
forecast_df["model_version"] = champion_version.version

# Bring category/store labels along for convenience -- the UI and GenAI layer
# shouldn't need to join back to the dimension tables just to show a category name.
forecast_df = forecast_df.merge(
    combo_static[["product_id", "store_id", "category", "brand"]], on=["product_id", "store_id"], how="left"
).merge(
    stores_pd[["store_id", "store_name", "city"]], on="store_id", how="left"
)

forecast_columns = [
    "forecast_id", "forecast_date", "product_id", "category", "brand",
    "store_id", "store_name", "city", "forecast_horizon",
    "predicted_units", "prediction_timestamp", "model_name", "model_version",
]
forecast_df = forecast_df[forecast_columns].sort_values(["product_id", "store_id", "forecast_date"])

display(forecast_df.head(20))

# COMMAND ----------

forecast_spark_df = spark.createDataFrame(forecast_df)

(
    forecast_spark_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.forecast_results")
)

print(f"Forecast written to {CATALOG}.{GOLD_SCHEMA}.forecast_results ({len(forecast_df):,} rows).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity Check — Forecast Trend by Category
# MAGIC
# MAGIC A quick visual check that the 30-day forecast looks like a plausible
# MAGIC continuation of history, not a discontinuity or a flat line (a flat
# MAGIC line across all 30 days would suggest the recursive loop isn't
# MAGIC actually varying its inputs day to day).

# COMMAND ----------

import matplotlib.pyplot as plt

category_forecast = forecast_df.groupby(["forecast_date", "category"])["predicted_units"].sum().reset_index()

fig, ax = plt.subplots(figsize=(11, 5))
for category, group in category_forecast.groupby("category"):
    ax.plot(group["forecast_date"], group["predicted_units"], marker="o", markersize=3, label=category)
ax.set_title(f"30-Day Forecast by Category (from {anchor_date.date()})")
ax.set_ylabel("Predicted units sold (all stores)")
ax.legend(fontsize=8, ncol=2)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

print("Batch inference completed successfully.")
print(f"Model: {MODEL_NAME} version {champion_version.version}")
print(f"Forecast window: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
print(f"Rows written: {len(forecast_df):,}")