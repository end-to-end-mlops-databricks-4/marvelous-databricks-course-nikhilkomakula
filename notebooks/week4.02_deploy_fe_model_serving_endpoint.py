# Databricks notebook source
# MAGIC %pip install hotel_reservation-0.0.1-py3-none-any.whl

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

from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.fe_model_serving import FeatureLookupServing

# COMMAND ----------


spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value


# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "hotel-reservation-model-serving-fe"

# COMMAND ----------

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.hotel_features",
)

# COMMAND ----------

fe = FeatureEngineeringClient()
online_store_name = "hotel-reservation-predictions"

# COMMAND ----------

fe.get_online_store(name=online_store_name)

# COMMAND ----------

# fe.delete_online_store(name=online_store_name)

# COMMAND ----------

# Create online store
if fe.get_online_store(name=online_store_name) is None:
    fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"
    )
    online_store = fe.get_online_store(name=online_store_name)
else:
    online_store = fe.get_online_store(name=online_store_name)

# COMMAND ----------

# Create the online table for hotel reservations
# feature_model_server.create_online_table(online_store=online_store)
feature_model_server.create_or_update_online_table(online_store_name=online_store_name)

# COMMAND ----------

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Create a sample request body
required_columns = [
    "Booking_ID",
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "market_segment_type",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests"
]

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])


# COMMAND ----------

dataframe_records[0]

# COMMAND ----------

# Call the endpoint with one sample record
def call_endpoint(record) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

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