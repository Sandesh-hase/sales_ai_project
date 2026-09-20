# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Bronze Layer — Data Ingestion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC This notebook loads raw CSV files from the Unity Catalog Volume into
# MAGIC Bronze Delta tables. It handles two different patterns:
# MAGIC
# MAGIC - **`products` / `stores`** — static reference data. Only reloaded (full
# MAGIC   overwrite) if a fresh file is found at the root of the Volume; if not
# MAGIC   present, the existing Bronze table is left untouched. These are not
# MAGIC   expected to change on every pipeline run.
# MAGIC - **`sales_transactions` / `calendar_marketing_external`** — incremental
# MAGIC   data. New files are dropped into an `incoming/` subfolder (any number
# MAGIC   of files, any naming as long as it starts with the right prefix) and
# MAGIC   are **merged (upserted)** into Bronze rather than overwritten, so
# MAGIC   re-running this notebook is always safe. Processed files are moved to
# MAGIC   `incoming/processed/` afterward so they aren't picked up again.
# MAGIC
# MAGIC At real scale, this "landing zone + archive" pattern is typically
# MAGIC replaced by Databricks **Auto Loader** (`cloudFiles`), which tracks
# MAGIC processed files via a managed checkpoint instead of a manual archive
# MAGIC folder. The pattern here is functionally equivalent and easier to
# MAGIC follow/demo, which is why it's used for this project.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration

# COMMAND ----------

CATALOG = "sales_ai"
BRONZE_SCHEMA = "bronze"

RAW_PATH = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw_files"
INCOMING_PATH = f"{RAW_PATH}/incoming"
PROCESSED_PATH = f"{INCOMING_PATH}/processed"

# Static reference files -- root of the volume, only reloaded if present
PRODUCTS_FILE = f"{RAW_PATH}/products.csv"
STORES_FILE = f"{RAW_PATH}/stores.csv"

# Incremental file prefixes -- any file in incoming/ starting with these is picked up
SALES_FILE_PREFIX = "sales_transactions"
CALENDAR_FILE_PREFIX = "calendar_marketing_external"

# Ensure the landing-zone folders exist so dbutils.fs.ls doesn't fail on a fresh setup
dbutils.fs.mkdirs(INCOMING_PATH)
dbutils.fs.mkdirs(PROCESSED_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Functions

# COMMAND ----------

def read_csv_file(path):
    """Read a single CSV (or a list of CSV paths) from the volume."""
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(path)
    )

def standardize_columns(df):
    """Convert column names to lowercase snake_case."""
    for column in df.columns:
        new_column = column.strip().lower().replace(" ", "_").replace("-", "_")
        df = df.withColumnRenamed(column, new_column)
    return df

def file_exists(path):
    """Check whether a single file exists in the volume without raising an error."""
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

def discover_incoming_files(prefix):
    """
    Return the full paths of every file in incoming/ whose name starts with
    `prefix` and ends in .csv. Returns an empty list if none are found --
    callers should treat that as "nothing new to process", not an error.
    """
    all_files = dbutils.fs.ls(INCOMING_PATH)
    matches = [
        f.path for f in all_files
        if f.name.startswith(prefix) and f.name.endswith(".csv")
    ]
    return matches

def merge_into_bronze(new_df, table_name, merge_key):
    """
    Upsert new_df into a Bronze Delta table on merge_key. Creates the table
    on first run if it doesn't exist yet; merges (update-if-matched,
    insert-if-not) on every subsequent run. This makes incremental loads
    idempotent -- re-running on the same file twice does not create
    duplicate rows.
    """
    full_table_name = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"

    if not spark.catalog.tableExists(full_table_name):
        (
            new_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(full_table_name)
        )
        print(f"[{table_name}] Table did not exist -- created with {new_df.count():,} rows.")
        return

    target = DeltaTable.forName(spark, full_table_name)
    (
        target.alias("t")
        .merge(new_df.alias("s"), f"t.{merge_key} = s.{merge_key}")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"[{table_name}] Merged {new_df.count():,} incoming rows on '{merge_key}'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Static Reference Data (Products / Stores)
# MAGIC
# MAGIC Only processed if the corresponding file is present at the root of the
# MAGIC volume. This is expected to be true on the very first run (initial
# MAGIC catalog load) and false on later incremental runs where only new sales
# MAGIC and campaign data is uploaded -- in that case the existing Bronze
# MAGIC tables for products/stores are left exactly as they are.

# COMMAND ----------

if file_exists(PRODUCTS_FILE):
    products_df = standardize_columns(read_csv_file(PRODUCTS_FILE))
    (
        products_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{BRONZE_SCHEMA}.products")
    )
    print(f"[products] Reloaded from {PRODUCTS_FILE} -- {products_df.count():,} rows.")
else:
    print(f"[products] No file found at {PRODUCTS_FILE} -- leaving existing Bronze table unchanged.")

if file_exists(STORES_FILE):
    stores_df = standardize_columns(read_csv_file(STORES_FILE))
    (
        stores_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{BRONZE_SCHEMA}.stores")
    )
    print(f"[stores] Reloaded from {STORES_FILE} -- {stores_df.count():,} rows.")
else:
    print(f"[stores] No file found at {STORES_FILE} -- leaving existing Bronze table unchanged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Incremental Data (Sales Transactions)
# MAGIC
# MAGIC Discovers every matching file in `incoming/` -- one file or many are
# MAGIC handled identically, since Spark unions a list of CSV paths into a
# MAGIC single DataFrame automatically. Rows are merged into Bronze on
# MAGIC `transaction_id`, then successfully processed files are archived.

# COMMAND ----------

sales_files = discover_incoming_files(SALES_FILE_PREFIX)

if sales_files:
    print(f"[sales_transactions] Found {len(sales_files)} new file(s):")
    for f in sales_files:
        print(f"  - {f}")

    sales_df = standardize_columns(read_csv_file(sales_files))
    merge_into_bronze(sales_df, "sales_transactions", merge_key="transaction_id")

    for f in sales_files:
        dest = f.replace(INCOMING_PATH, PROCESSED_PATH)
        dbutils.fs.mv(f, dest)
    print(f"[sales_transactions] Archived {len(sales_files)} file(s) to {PROCESSED_PATH}.")
else:
    print(f"[sales_transactions] No new files found in {INCOMING_PATH}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Incremental Data (Calendar / Marketing / External)
# MAGIC
# MAGIC Same pattern as sales, merged on `date`.

# COMMAND ----------

calendar_files = discover_incoming_files(CALENDAR_FILE_PREFIX)

if calendar_files:
    print(f"[calendar_marketing_external] Found {len(calendar_files)} new file(s):")
    for f in calendar_files:
        print(f"  - {f}")

    calendar_df = standardize_columns(read_csv_file(calendar_files))
    merge_into_bronze(calendar_df, "calendar_marketing_external", merge_key="date")

    for f in calendar_files:
        dest = f.replace(INCOMING_PATH, PROCESSED_PATH)
        dbutils.fs.mv(f, dest)
    print(f"[calendar_marketing_external] Archived {len(calendar_files)} file(s) to {PROCESSED_PATH}.")
else:
    print(f"[calendar_marketing_external] No new files found in {INCOMING_PATH}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Bronze Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN sales_ai.bronze;

# COMMAND ----------

for table in ["sales_transactions", "products", "stores", "calendar_marketing_external"]:
    count = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.{table}").count()
    print(f"{table:30s}: {count:,} rows")

# COMMAND ----------

print("Bronze ingestion completed successfully.")