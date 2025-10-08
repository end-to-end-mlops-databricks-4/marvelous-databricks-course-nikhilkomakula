# Databricks notebook source
# MAGIC %pip install hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import time

import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.model_serving import ModelServing

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservation_model_basic", endpoint_name="hotel-reservation-model-serving"
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Create a sample request body
required_columns = [
    # "Booking_ID",
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

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

dataframe_records[0]

# COMMAND ----------

# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'no_of_adults': 1,
  'no_of_children': 0,
  'no_of_weekend_nights': 2,
  'no_of_week_nights': 1,
  'type_of_meal_plan': 'Meal Plan 1',
  'required_car_parking_space': 0,
  'room_type_reserved': 'Room_Type 1',
  'lead_time': 116,
  'arrival_year': 2018,
  'arrival_month': 2,
  'arrival_date': 28,
  'market_segment_type': 'Offline',
  'repeated_guest': 0,
  'no_of_previous_cancellations': 0,
  'no_of_previous_bookings_not_canceled': 0,
  'avg_price_per_room': 76.0,
  'no_of_special_requests': 0}]
"""

def call_endpoint(record) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/hotel-reservation-model-serving/invocations"

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

# COMMAND ----------

