# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the hotel reservations dataset
spark = SparkSession.builder.getOrCreate()

filepath = "../data/Hotel Reservations.csv"

# Load the data
df = pd.read_csv(filepath)


# COMMAND ----------
# Load the hotel reservations dataset

data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

logger.info("Data preprocessing is completed.")

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
# COMMAND ----------
