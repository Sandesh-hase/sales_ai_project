# Databricks notebook source
# MAGIC %md
# MAGIC # Sales AI — Databricks Environment Setup
# MAGIC
# MAGIC This notebook initializes the Databricks environment for the **AI-Powered Retail Sales Forecasting & Business Decision Support Platform**.
# MAGIC
# MAGIC ### Objectives
# MAGIC
# MAGIC - Create the project catalog
# MAGIC - Create Medallion Architecture schemas
# MAGIC - Create a Volume for raw CSV files
# MAGIC - Verify the created resources

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Project Configuration
# MAGIC
# MAGIC The project uses the following naming convention:
# MAGIC
# MAGIC - **Project:** `sales-ai`
# MAGIC - **Catalog:** `sales_ai`
# MAGIC - **Schemas:** `bronze`, `silver`, `gold`
# MAGIC - **Raw Volume:** `raw_files`
# MAGIC
# MAGIC The catalog represents the overall project, while the schemas represent the different stages of the data lifecycle.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the project catalog.
# MAGIC -- IF NOT EXISTS makes this command safe to execute multiple times.
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS sales_ai;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use the project catalog for all subsequent operations.
# MAGIC
# MAGIC USE CATALOG sales_ai;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Medallion Architecture Schemas
# MAGIC
# MAGIC We will use the standard Medallion Architecture:
# MAGIC
# MAGIC ### Bronze
# MAGIC Raw ingested data with minimal transformation.
# MAGIC
# MAGIC ### Silver
# MAGIC Cleaned, validated, standardized, and integrated data.
# MAGIC
# MAGIC ### Gold
# MAGIC Business-ready and ML-ready datasets, including engineered features.

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS silver;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS gold;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Raw Data Volume
# MAGIC
# MAGIC The raw CSV files will be uploaded to a Unity Catalog Volume.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a governed Volume for raw project files.
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS sales_ai.bronze.raw_files;

# COMMAND ----------

