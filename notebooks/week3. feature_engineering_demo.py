# Databricks notebook source
# install dependencies
# %pip install -e ..
# %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# MAGIC %pip install hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

#restart python
%restart_python


# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow

from hotel_reservation.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from hotel_reservation.utils import is_databricks
from dotenv import load_dotenv
import os
from mlflow import MlflowClient
import pandas as pd
from hotel_reservation import __version__
from mlflow.utils.environment import _mlflow_conda_env
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.errors import AnalysisException
import numpy as np
from datetime import datetime
import boto3


# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------

# create feature table with information about hotel reservations

feature_table_name = f"{config.catalog_name}.{config.schema_name}.hotel_reservation_features_demo"
lookup_features = ["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"]


# COMMAND ----------

# Option 1: feature engineering client
feature_table = fe.create_table(
   name=feature_table_name,
   primary_keys=["Booking_ID"],
   df=train_set[["Booking_ID"]+lookup_features],
   description="Hotel Reservation features table",
)

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

fe.write_table(
   name=feature_table_name,
   df=test_set[["Booking_ID"]+lookup_features],
   mode="merge",
)

# COMMAND ----------

# # create feature table with information about hotel reservations
# # Option 2: SQL

# spark.sql(f"""
#           CREATE OR REPLACE TABLE {feature_table_name}
#           (Booking_ID STRING NOT NULL, no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT);
#           """)
# # primary key on Databricks is not enforced!
# try:
#     spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT hotel_reservation_pk_demo PRIMARY KEY(Booking_ID);")
# except AnalysisException:
#     pass
# spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
# spark.sql(f"""
#           INSERT INTO {feature_table_name}
#           SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
#           FROM {config.catalog_name}.{config.schema_name}.train_set
#           """)
# spark.sql(f"""
#           INSERT INTO {feature_table_name}
#           SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
#           FROM {config.catalog_name}.{config.schema_name}.test_set
#           """)

# COMMAND ----------

# create feature function
# docs: https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function

# problems with feature functions:
# functions are not versioned 
# functions may behave differently depending on the runtime (and version of packages and python)
# there is no way to enforce python version & package versions for the function 
# this is only supported from runtime 17
# advised to use only for simple calculations

function_name = f"{config.catalog_name}.{config.schema_name}.calculate_booking_value_demo"

# COMMAND ----------


# Option 1: with Python
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}(
                avg_price_per_room DOUBLE,
                no_of_weekend_nights BIGINT,
                no_of_week_nights BIGINT
        )
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        if avg_price_per_room is None or no_of_weekend_nights is None or no_of_week_nights is None:
                return None
        return avg_price_per_room * (no_of_weekend_nights + no_of_week_nights)
        $$
""")

# COMMAND ----------

# # it is possible to define simple functions in sql only without python
# # Option 2
# spark.sql(f"""
#         CREATE OR REPLACE FUNCTION {function_name}_sql(
#                 avg_price_per_room DOUBLE,
#                 no_of_weekend_nights INT,
#                 no_of_week_nights INT
#         )
#         RETURNS DOUBLE
#         RETURN 
#         CASE 
#                 WHEN avg_price_per_room IS NULL 
#                         OR no_of_weekend_nights IS NULL 
#                         OR no_of_week_nights IS NULL 
#                 THEN NULL
#                 ELSE avg_price_per_room * (no_of_weekend_nights + no_of_week_nights)
#         END

#         """)

# COMMAND ----------

# # Run the query and get the result as a DataFrame
# result_df = spark.sql(f"SELECT {function_name}_sql(90.1, 2, 2) AS booking_value")

# # Show the result in tabular form
# result_df.show()

# COMMAND ----------

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
                table_name=feature_table_name,
                feature_names=["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"],
                lookup_key="Booking_ID",
            ),
        FeatureFunction(
                udf_name=function_name,
                output_name="booking_value",
                input_bindings={"avg_price_per_room": "avg_price_per_room", "no_of_weekend_nights": "no_of_weekend_nights", "no_of_week_nights": "no_of_week_nights"},
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.num_features + config.cat_features + ["booking_value"]]
y_train = training_df[config.target]

# COMMAND ----------

# Encode the label
labelEncoder = LabelEncoder()
y_train_encoded = labelEncoder.fit_transform(y_train)

# COMMAND ----------

pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("classifier", LGBMClassifier(**config.parameters))]
        )

pipeline.fit(X_train, y_train_encoded)

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
    

# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

# make predictions
features = [f for f in ["Booking_ID"] + config.num_features + config.cat_features if f not in lookup_features]
predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set[features]
)

# COMMAND ----------

predictions.select("prediction").show(5)

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Booking_ID"] + config.num_features + config.cat_features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features)
# .withColumn(
#     "Booking_ID",
#     (col("Booking_ID").cast("long") + 1000000).cast("string")
# )

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id 
)

# COMMAND ----------

# make predictions for a non-existing entry -> error!
predictions.select("prediction").show(5)

# COMMAND ----------

no_of_previous_cancellations_function = f"{config.catalog_name}.{config.schema_name}.replace_no_of_previous_cancellations_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {no_of_previous_cancellations_function}(no_of_previous_cancellations BIGINT)
        RETURNS BIGINT
        LANGUAGE PYTHON AS
        $$
        if no_of_previous_cancellations is None:
            return 0
        else:
            return no_of_previous_cancellations
        $$
        """)

no_of_previous_bookings_not_canceled_function = f"{config.catalog_name}.{config.schema_name}.replace_no_of_previous_bookings_not_canceled_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {no_of_previous_bookings_not_canceled_function}(no_of_previous_bookings_not_canceled BIGINT)
        RETURNS BIGINT
        LANGUAGE PYTHON AS
        $$
        if no_of_previous_bookings_not_canceled is None:
            return 1
        else:
            return no_of_previous_bookings_not_canceled
        $$
        """)

# COMMAND ----------

# what if we want to replace with a default value if entry is not found
# what if we want to look up value in another table? the logics get complex
# problems that arize: functions/ lookups always get executed (if statememt is not possible)
# it can get slow...

# step 1: create 3 feature functions

# step 2: redefine create training set

# try again

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"],
            lookup_key="Booking_ID",
            rename_outputs={"no_of_previous_cancellations": "lookup_no_of_previous_cancellations",
                            "no_of_previous_bookings_not_canceled": "lookup_no_of_previous_bookings_not_canceled"}
                ),
        FeatureFunction(
            udf_name=no_of_previous_cancellations_function,
            output_name="no_of_previous_cancellations",
            input_bindings={"no_of_previous_cancellations": "lookup_no_of_previous_cancellations"},
            ),
        FeatureFunction(
            udf_name=no_of_previous_bookings_not_canceled_function,
            output_name="no_of_previous_bookings_not_canceled",
            input_bindings={"no_of_previous_bookings_not_canceled": "lookup_no_of_previous_bookings_not_canceled"},
        ),
        FeatureFunction(
                udf_name=function_name,
                output_name="booking_value",
                input_bindings={"avg_price_per_room": "avg_price_per_room", "no_of_weekend_nights": "no_of_weekend_nights", "no_of_week_nights": "no_of_week_nights"},
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.num_features + config.cat_features + ["booking_value"]]
y_train = training_df[config.target]
y_train_encoded = labelEncoder.fit_transform(y_train)

#pipeline
pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("classifier", LGBMClassifier(**config.parameters))]
        )

pipeline.fit(X_train, y_train_encoded)

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Booking_ID"] + config.num_features + config.cat_features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features)
# .withColumn(
#     "Booking_ID",
#     (col("Id").cast("long") + 1000000).cast("string")
# )

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id 
)

# COMMAND ----------

# make predictions for a non-existing entry -> no error!
predictions.select("prediction").show(5)

# COMMAND ----------

dbutils.secrets.get(scope="mlops", key="aws_access_key_id")

# COMMAND ----------

dbutils.secrets.get(scope="mlops", key="aws_secret_access_key")

# COMMAND ----------

dbutils.secrets.list("mlops")

# COMMAND ----------

import boto3

region_name = "eu-west-1"

client = boto3.client(
    'dynamodb',
    aws_access_key_id=dbutils.secrets.get(scope="mlops", key="aws_access_key_id"),
    aws_secret_access_key=dbutils.secrets.get(scope="mlops", key="aws_secret_access_key"),
    region_name=region_name
)

# COMMAND ----------

response = client.create_table(
    TableName='HotelFeatures',
    KeySchema=[
        {
            'AttributeName': 'Booking_ID',
            'KeyType': 'HASH'  # Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'Booking_ID',
            'AttributeType': 'S'  # String
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Table creation initiated:", response['TableDescription']['TableName'])

# COMMAND ----------

client.put_item(
    TableName='HotelFeatures',
    Item={
        'Booking_ID': {'S': 'hotel_001'},
        'no_of_previous_cancellations': {'N': '8'},
        'no_of_previous_bookings_not_canceled': {'N': '2450'}
    }
)

# COMMAND ----------

response = client.get_item(
    TableName='HotelFeatures',
    Key={
        'Booking_ID': {'S': 'hotel_001'}
    }
)

# Extract the item from the response
item = response.get('Item')
print(item)

# COMMAND ----------

from itertools import islice

rows = spark.table(feature_table_name).toPandas().to_dict(orient="records")

def to_dynamodb_item(row):
    return {
        'PutRequest': {
            'Item': {
                'Booking_ID': {'S': str(row['Booking_ID'])},
                'no_of_previous_cancellations': {'N': str(row['no_of_previous_cancellations'])},
                'no_of_previous_bookings_not_canceled': {'N': str(row['no_of_previous_bookings_not_canceled'])}
            }
        }
    }

items = [to_dynamodb_item(row) for row in rows]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunks(items, 25):
    response = client.batch_write_item(
        RequestItems={
            'HotelFeatures': batch
        }
    )
    # Handle any unprocessed items if needed
    unprocessed = response.get('UnprocessedItems', {})
    if unprocessed:
        print("Warning: Some items were not processed. Retry logic needed.")

# COMMAND ----------

# We ran into more limitations when we tried complex data types as output of a feature function
# and then tried to use it for serving
# al alternatve solution: using an external database (we use DynamoDB here)

# create a DynamoDB table
# insert records into dynamo DB & read from dynamoDB

# create a pyfunc model

# COMMAND ----------


class HotelReservationModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting hotel reservation's booking status.
    """

    def __init__(self, model: object) -> None:
        """Initialize the HotelReservationModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> dict[str, float]:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the adjusted prediction.
        """
        client = boto3.client('dynamodb',
                                   aws_access_key_id=os.environ["aws_access_key_id"],
                                   aws_secret_access_key=os.environ["aws_access_key"],
                                   region_name=region_name)
        
        parsed = []
        for lookup_id in model_input["Booking_ID"]:
            raw_item = client.get_item(
                TableName='HotelFeatures',
                Key={'Booking_ID': {'S': lookup_id}})["Item"]     
            parsed_dict = {key: int(value['N']) if 'N' in value else value['S']
                      for key, value in raw_item.items()}
            parsed.append(parsed_dict)
        lookup_df=pd.DataFrame(parsed)
        merged_df = model_input.merge(lookup_df, on="Booking_ID", how="left").drop("Booking_ID", axis=1)
        
        merged_df["no_of_previous_cancellations"] = merged_df["no_of_previous_cancellations"].fillna(2)
        merged_df["no_of_previous_bookings_not_canceled"] = merged_df["no_of_previous_bookings_not_canceled"].fillna(2)
        merged_df["booking_value"] = merged_df["avg_price_per_room"] * (merged_df["no_of_weekend_nights"] + merged_df["no_of_week_nights"])
        predictions = self.model.predict(merged_df)

        return [int(x) for x in predictions]

# COMMAND ----------

custom_model = HotelReservationModelWrapper(pipeline)

# COMMAND ----------

features = [f for f in ["Booking_ID"] + config.num_features + config.cat_features if f not in lookup_features]
data = test_set.select(*features).toPandas()
data

# COMMAND ----------

custom_model.predict(context=None, model_input=data)

# COMMAND ----------

#log model
mlflow.set_experiment("/Shared/demo-model-fe-pyfunc")
with mlflow.start_run(run_name="demo-run-model-fe-pyfunc",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=data, model_output=custom_model.predict(context=None, model_input=data))
    mlflow.pyfunc.log_model(
                python_model=custom_model,
                artifact_path="lightgbm-pipeline-model-fe-custom",
                signature=signature,
            )
    

# COMMAND ----------

# predict
mlflow.models.predict(f"runs:/{run_id}/lightgbm-pipeline-model-fe", data[0:1])