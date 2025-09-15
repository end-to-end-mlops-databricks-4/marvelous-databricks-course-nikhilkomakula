# Databricks notebook source

import json
import os

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from hotel_reservation import __version__
from hotel_reservation.config import ProjectConfig
from hotel_reservation.utils import adjust_predictions, is_databricks

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
X_train = train_set[config.num_features + config.cat_features]
y_train = train_set[config.target]

# COMMAND ----------
X_train.head()

# COMMAND ----------
X_train.columns.to_list()

# COMMAND ----------
y_train.head()

# COMMAND ----------
print(config.parameters)

# COMMAND ----------
# Apply LabelEncoder to target column
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# COMMAND ----------
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), config.cat_features)],
                remainder="passthrough",
            ),
        ),
        ("classifier", LGBMClassifier(**config.parameters)),
    ]
)

pipeline.fit(X_train, y_train_encoded)

# COMMAND ----------
mlflow.set_experiment("/Users/nikhil.komakula@gmail.com/hotel_reservation")
with mlflow.start_run(
    run_name="hotel-reservation-run-model",
    tags={"git_sha": "1234567890abcd", "branch": "week2"},
    description="hotel reservation run for model logging",
) as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    model_info = mlflow.sklearn.log_model(sk_model=pipeline, name="lightgbm-pipeline-model", signature=signature)

# COMMAND ----------
# Load the model using the alias and test predictions - not recommended!
# This may be working in a notebook but will fail on the endpoint
artifact_uri = mlflow.get_run(run_id=run_id).to_dictionary()["info"]["artifact_uri"]

# COMMAND ----------
logged_model = mlflow.get_logged_model(model_info.model_id)

# COMMAND ----------
# two ways of loading the model
# using model id
model = mlflow.sklearn.load_model(f"models:/{model_info.model_id}")

# COMMAND ----------
# using run id
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------

logged_model_dict = logged_model.to_dictionary()
logged_model_dict["metrics"] = [x.__dict__ for x in logged_model_dict["metrics"]]
with open("../demo_artifacts/logged_model.json", "w") as json_file:
    json.dump(logged_model_dict, json_file, indent=4)

# COMMAND ----------
print(logged_model.params)

# COMMAND ----------
print(logged_model.metrics)

# COMMAND ----------
model_name = f"{config.catalog_name}.{config.schema_name}.hotel_reservation"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"git_sha": "1234567890abcd"}
)

# COMMAND ----------
# only searching by name is supported
v = mlflow.search_model_versions(filter_string=f"name='{model_name}'")
print(v[0].__dict__)

# COMMAND ----------
# not supported
mlflow.search_model_versions(filter_string=f"run_id='{run_id}'")

# COMMAND ----------
# not supported
v = mlflow.search_model_versions(filter_string="tags.git_sha='1234567890abcd'")

# COMMAND ----------
client = MlflowClient()

# COMMAND ----------
# this will fail: latest is reserved
client.set_registered_model_alias(name=model_name, alias="latest", version=model_version.version)

# COMMAND ----------
# loading latest also fails
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@latest")

# COMMAND ----------
# let's set latest-model alias instead
client.set_registered_model_alias(name=model_name, alias="latest-model", version=model_version.version)

# COMMAND ----------
model_uri = f"models:/{model_name}@latest-model"
sklearn_pipeline = mlflow.sklearn.load_model(model_uri)
predictions = sklearn_pipeline.predict(X_train[0:1])
print(predictions)

# COMMAND ----------
print(label_encoder.inverse_transform(predictions))

# COMMAND ----------
print(y_train[0:1])

# COMMAND ----------
probabilities = sklearn_pipeline.predict_proba(X_train[0:1])
print(probabilities)

# COMMAND ----------
# A better way, also explained here
#  will work in a later version of mlflow:
# https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-debug
# https://www.databricksters.com/p/pyfunc-it-well-do-it-live

# mlflow.models.predict(model_uri, X_train[0:1])

# COMMAND ----------
# Let's wrap it around a custom model


class HotelReservationModelWrapper(mlflow.pyfunc.PythonModel):
    """A custom MLflow PythonModel wrapper for hotel reservation predictions.

    This wrapper applies a custom prediction threshold to model outputs.
    """

    def __init__(self, model: BaseEstimator) -> None:
        """Initialize the model wrapper."""
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame, threshold: float = 0.5
    ) -> dict[str, np.ndarray]:
        """Generate predictions with a custom threshold applied."""
        if isinstance(model_input, pd.DataFrame):
            # Get probabilities for the positive class (binary classification)
            probabilities = self.model.predict_proba(model_input)[:, 1]

            # Apply custom threshold
            adjusted_preds = adjust_predictions(probabilities, threshold=threshold)

            return {"Prediction": adjusted_preds}
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
wrapped_model = HotelReservationModelWrapper(sklearn_pipeline)  # we pass the loaded model to the wrapper

mlflow.set_experiment(experiment_name="/Users/nikhil.komakula@gmail.com/hotel_reservation-pyfunc")
with mlflow.start_run(tags={"branch": "week2", "git_sha": "1234567890abcd"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": [0]})
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            f"code/hotel_reservation-{__version__}-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        name="pyfunc-hotel-reservation-model",
        code_paths=[f"../dist/hotel_reservation-{__version__}-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
# Another way of doing the same thing:


class HotelReservationModelWrapper2(mlflow.pyfunc.PythonModel):
    """A custom MLflow PythonModel wrapper for hotel reservation predictions.

    This wrapper applies a custom prediction threshold to model outputs.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the trained LightGBM pipeline model from MLflow artifacts.

        This method is called automatically by MLflow when the model is deployed
        or loaded for inference. It retrieves the serialized LightGBM pipeline
        stored in the model artifacts and assigns it to the wrapper's `self.model`.
        """
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline-model"])

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame, threshold: float = 0.5
    ) -> dict[str, np.ndarray]:
        """Generate predictions for hotel reservation cancellation using a custom threshold."""
        if isinstance(model_input, pd.DataFrame):
            # Get probabilities for the positive class (binary classification)
            probabilities = self.model.predict_proba(model_input)[:, 1]

            # Apply custom threshold
            adjusted_preds = adjust_predictions(probabilities, threshold=threshold)

            return {"Prediction": adjusted_preds}
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
mlflow.set_experiment(experiment_name="/Users/nikhil.komakula@gmail.com/hotel-reservation-pyfunc")
with mlflow.start_run(tags={"branch": "week2", "git_sha": "1234567890abcd"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": [0]})
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            f"code/hotel_reservation-{__version__}-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=HotelReservationModelWrapper2(),
        name="pyfunc-hotel-reservation-model",
        artifacts={"lightgbm-pipeline-model": f"models:/{model_name}@latest-model"},
        code_paths=[f"../dist/hotel_reservation-{__version__}-py3-none-any.whl"],
        signature=signature,
    )
# COMMAND ----------
print(run_id)

# COMMAND ----------
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-hotel-reservation-model")

# COMMAND ----------
pyfunc_model.predict(X_train[0:1])

# COMMAND ----------
