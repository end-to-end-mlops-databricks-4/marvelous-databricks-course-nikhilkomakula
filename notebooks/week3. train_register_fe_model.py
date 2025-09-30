# Databricks notebook source
# MAGIC %pip install hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Configure tracking uri
from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")


# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define booking value feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# Train the model
fe_model.register_model()

# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("no_of_previous_cancellations", "no_of_previous_bookings_not_canceled", config.target)


# COMMAND ----------

X_test.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col

# X_test = X_test.withColumn("no_of_week_nights", col("no_of_week_nights").cast("int")).withColumn(
#     "no_of_weekend_nights", col("no_of_weekend_nights").cast("int")
# )

# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)

# COMMAND ----------

display(predictions)