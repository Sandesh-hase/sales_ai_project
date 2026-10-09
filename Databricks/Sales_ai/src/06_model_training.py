# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Sales AI — Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook trains **one global gradient boosting model** on
# MAGIC `sales_ai.gold.sales_features` — not one model per category. `category`,
# MAGIC `product_id`, and `store_id` are included as model features instead, so
# MAGIC the model still learns each series' distinct pattern while sharing
# MAGIC learning across series (e.g. "how Diwali affects demand" generalizes
# MAGIC across categories, rather than being relearned six times).
# MAGIC
# MAGIC What this notebook does, in order:
# MAGIC 1. Load the Gold feature table and split by the pre-computed `dataset_split` column (time-based, not random)
# MAGIC 2. Establish a **naive baseline** (last week's value) — the model only earns its place if it beats this
# MAGIC 3. Train a LightGBM regressor with early stopping on the validation set
# MAGIC 4. Evaluate on validation and test with MAE / RMSE / MAPE / WAPE
# MAGIC 5. Log everything to MLflow and register the model in the Unity Catalog Model Registry
# MAGIC 6. Only promote the new model to the `Champion` alias if it beats the current Champion — otherwise it's registered as a candidate for manual review

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports and Configuration

# COMMAND ----------

!pip install lightgbm

# COMMAND ----------

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient

CATALOG = "sales_ai"
GOLD_SCHEMA = "gold"

MODEL_NAME = f"{CATALOG}.{GOLD_SCHEMA}.sales_forecasting_lgbm"
EXPERIMENT_NAME = "/Shared/sales_ai_forecasting"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Gold Feature Table
# MAGIC
# MAGIC The table is small enough (well under 100K rows) to bring into a
# MAGIC single-node pandas DataFrame for training. For a materially larger
# MAGIC catalog, this step would move to Spark-distributed training instead
# MAGIC (e.g. `lightgbm.spark` or a partitioned training approach).

# COMMAND ----------

gold_df = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.sales_features")
df = gold_df.toPandas()
df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")
print(f"Date range : {df['date'].min().date()} to {df['date'].max().date()}")
print(df["dataset_split"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature and Target Definition
# MAGIC
# MAGIC Categorical columns are cast to pandas `category` dtype so LightGBM
# MAGIC handles them natively (splits directly on category, no manual
# MAGIC one-hot-encoding needed). Nulls in both numeric and categorical
# MAGIC features are left as-is — LightGBM learns the best split direction for
# MAGIC missing values rather than requiring an imputed value.
# MAGIC
# MAGIC Calendar fields (`month`, `day_of_week`, `quarter`, `week_of_year`) are
# MAGIC kept as plain integers rather than cyclically encoded (sin/cos): that
# MAGIC encoding mainly helps linear and neural models, which assume a smooth
# MAGIC numeric relationship. A tree-based model like LightGBM splits on
# MAGIC thresholds directly, so it doesn't need the cyclical transform to learn
# MAGIC "December is close to January" — it learns that from the data itself.

# COMMAND ----------

TARGET_COLUMN = "units_sold"

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

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype("category")

for col in BOOLEAN_FEATURES:
    df[col] = df[col].astype("float64")  # True/False/NULL -> 1.0/0.0/NaN, LightGBM-friendly

print(f"Feature count: {len(FEATURE_COLUMNS)}  ({len(CATEGORICAL_FEATURES)} categorical, "
      f"{len(NUMERIC_FEATURES)} numeric, {len(BOOLEAN_FEATURES)} boolean)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Validation / Test Split
# MAGIC
# MAGIC Uses the `dataset_split` column computed in the Gold notebook
# MAGIC (2021-2023 train, 2024 validation, 2025 test) — a time-based split, not
# MAGIC a random one, since this is a forecasting problem and the model must
# MAGIC never be validated on data from before what it was trained on.

# COMMAND ----------

train_df = df[df["dataset_split"] == "train"]
val_df = df[df["dataset_split"] == "validation"]
test_df = df[df["dataset_split"] == "test"]

X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

print(f"Train: {len(X_train):,} rows  |  Validation: {len(X_val):,} rows  |  Test: {len(X_test):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper — Forecast Accuracy Metrics
# MAGIC
# MAGIC **WAPE** (Weighted Absolute Percentage Error) is used as the primary
# MAGIC metric — unlike MAPE, it doesn't blow up or divide by zero on the
# MAGIC genuine zero-sales days this dataset intentionally contains, and it
# MAGIC weights errors by sales volume, which better reflects business impact.

# COMMAND ----------

def compute_metrics(y_true, y_pred):
    """
    Compute standard forecast accuracy metrics.
    MAPE is calculated only over rows where y_true > 0, since it is
    undefined at y_true == 0; WAPE is reported as the primary metric
    because it remains well-defined across zero-sales days.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)  # forecasted units sold cannot be negative

    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    nonzero_mask = y_true > 0
    mape = np.mean(np.abs(errors[nonzero_mask]) / y_true[nonzero_mask]) * 100 if nonzero_mask.any() else np.nan

    wape = np.sum(np.abs(errors)) / np.sum(y_true) * 100 if np.sum(y_true) > 0 else np.nan

    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2), "WAPE": round(wape, 2)}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Naive Baseline
# MAGIC
# MAGIC The simplest possible forecast: predict that today's sales will equal
# MAGIC last week's sales for the same series (`lag_7`). Any model trained
# MAGIC below is only worth deploying if it clears this bar by a meaningful
# MAGIC margin — a baseline this cheap sets the right expectations for what
# MAGIC "good accuracy" actually means here.

# COMMAND ----------

baseline_val_pred = val_df["lag_7"].fillna(val_df["lag_7"].median())
baseline_test_pred = test_df["lag_7"].fillna(test_df["lag_7"].median())

baseline_val_metrics = compute_metrics(y_val, baseline_val_pred)
baseline_test_metrics = compute_metrics(y_test, baseline_test_pred)

print("Naive baseline (lag_7) -- Validation:", baseline_val_metrics)
print("Naive baseline (lag_7) -- Test      :", baseline_test_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train LightGBM Model
# MAGIC
# MAGIC Early stopping on the validation set prevents the model from
# MAGIC overfitting to the training period — training halts once validation
# MAGIC error stops improving, rather than running for a fixed number of
# MAGIC rounds regardless of whether it's still helping.

# COMMAND ----------

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
}

model = lgb.LGBMRegressor(**LGBM_PARAMS)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="mae",
    categorical_feature=CATEGORICAL_FEATURES,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
)

print(f"Best iteration: {model.best_iteration_}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Against the Baseline

# COMMAND ----------

val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
test_pred = model.predict(X_test, num_iteration=model.best_iteration_)

model_val_metrics = compute_metrics(y_val, val_pred)
model_test_metrics = compute_metrics(y_test, test_pred)

comparison = pd.DataFrame({
    "Naive Baseline (Validation)": baseline_val_metrics,
    "LightGBM (Validation)": model_val_metrics,
    "Naive Baseline (Test)": baseline_test_metrics,
    "LightGBM (Test)": model_test_metrics,
}).T

display(comparison)

wape_improvement = (baseline_test_metrics["WAPE"] - model_test_metrics["WAPE"]) / baseline_test_metrics["WAPE"] * 100
print(f"\nWAPE improvement over naive baseline (test set): {wape_improvement:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance

# COMMAND ----------

importance_df = (
    pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
    .sort_values("importance", ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#1F8A55")
ax.set_title("Top 20 Feature Importances (LightGBM, split count)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Run, Model, and Metrics to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="lightgbm_global_model") as run:
    mlflow.log_params(LGBM_PARAMS)
    mlflow.log_param("best_iteration", model.best_iteration_)
    mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("validation_rows", len(X_val))
    mlflow.log_param("test_rows", len(X_test))

    for split_name, metrics in [("validation", model_val_metrics), ("test", model_test_metrics)]:
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{split_name}_{metric_name.lower()}", value)

    mlflow.log_metric("test_wape_improvement_pct", round(wape_improvement, 2))

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train.head(100)))
    model_info = mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id

print(f"MLflow run logged: {run_id}")
print(f"Model registered as: {MODEL_NAME}  (version {model_info.registered_model_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote to Champion — Only If It's Actually Better
# MAGIC
# MAGIC Unity Catalog uses model **aliases** rather than the older
# MAGIC stage-based registry. The newly trained version is only promoted to
# MAGIC the `Champion` alias if it beats the current Champion's test WAPE (or
# MAGIC if there isn't a Champion yet). Otherwise it's left registered as a
# MAGIC new version for manual review — a worse model is never silently
# MAGIC promoted just because it's the most recent one trained.

# COMMAND ----------

client = MlflowClient()
new_version = model_info.registered_model_version

current_champion_wape = None
try:
    champion_version = client.get_model_version_by_alias(MODEL_NAME, "Champion")
    champion_run = client.get_run(champion_version.run_id)
    current_champion_wape = champion_run.data.metrics.get("test_wape")
    print(f"Current Champion: version {champion_version.version}, test WAPE = {current_champion_wape}")
except Exception:
    print("No existing Champion found -- this will be the first Champion if it passes evaluation.")

should_promote = (current_champion_wape is None) or (model_test_metrics["WAPE"] < current_champion_wape)

if should_promote:
    client.set_registered_model_alias(MODEL_NAME, "Champion", new_version)
    print(f"Promoted version {new_version} to Champion (test WAPE = {model_test_metrics['WAPE']}).")
else:
    client.set_registered_model_alias(MODEL_NAME, "Challenger", new_version)
    print(f"Version {new_version} (test WAPE = {model_test_metrics['WAPE']}) did not beat the current "
          f"Champion (test WAPE = {current_champion_wape}). Registered as Challenger for manual review.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Predictions for the Dashboard / Next Notebook
# MAGIC
# MAGIC A validation + test prediction table is written to Gold so the next
# MAGIC notebook (batch inference / forecast generation) and the UI dashboard
# MAGIC have a ready reference of actual-vs-predicted for the historical
# MAGIC period, without needing to reload the model.

# COMMAND ----------

eval_results = pd.concat([
    val_df[["date", "product_id", "store_id", "units_sold"]].assign(predicted_units=val_pred, dataset_split="validation"),
    test_df[["date", "product_id", "store_id", "units_sold"]].assign(predicted_units=test_pred, dataset_split="test"),
])
eval_results["predicted_units"] = eval_results["predicted_units"].clip(lower=0).round(2)
eval_results["model_version"] = new_version
eval_results["mlflow_run_id"] = run_id

eval_results_spark = spark.createDataFrame(eval_results)

(
    eval_results_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.model_evaluation_results")
)

print(f"Evaluation results written to {CATALOG}.{GOLD_SCHEMA}.model_evaluation_results "
      f"({len(eval_results):,} rows).")

# COMMAND ----------

print("Model training pipeline completed successfully.")
print(f"Model: {MODEL_NAME}, version {new_version}")
print(f"Test WAPE: {model_test_metrics['WAPE']}%  (naive baseline: {baseline_test_metrics['WAPE']}%, "
      f"{wape_improvement:.1f}% improvement)")

# COMMAND ----------

