# Databricks notebook source
# MAGIC %pip install house_price-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time

import requests
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.serving.fe_model_serving import FeatureLookupServing

# COMMAND ----------


spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value


# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "house-prices-model-serving-fe"

# COMMAND ----------
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.house_prices_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.house_features",
)

# COMMAND ----------
# Create online store
fe = FeatureEngineeringClient()
online_store_name = "house-predictions"
if fe.get_online_store(name=online_store_name) is None:
    fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"
    )
    online_store = fe.get_online_store(name=online_store_name)
else:
    online_store = fe.get_online_store(name=online_store_name)
# COMMAND ----------

# Create the online table for house features
feature_model_server.create_online_table(online_store=online_store)

# COMMAND ----------
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "LotFrontage",
    "LotArea",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "TotalBsmtSF",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Neighborhood",
    "Condition1",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "SaleType",
    "SaleCondition",
    "Id",
]

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])


# COMMAND ----------
# Call the endpoint with one sample record
def call_endpoint(record) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
