# Databricks notebook source
!pip install hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

"""
Ray Tune + MLflow Nested Runs for Hotel Reservation Prediction
"""
import mlflow
import numpy as np
import pandas as pd
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from functools import partial
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.models.basic_model import BasicModel

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
tags = Tags(**{"git_sha": "abcd12345", "branch": "week3"})
spark = SparkSession.builder.getOrCreate()

basic_model = BasicModel(config=config, tags=tags, spark=spark)
basic_model.load_data()

# Split train set into train/validation for tuning
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    basic_model.X_train, basic_model.y_train_encoded, test_size=0.2, random_state=42
)

# COMMAND ----------

# --- Trainable function for Ray Tune with nested MLflow runs ---
""" This is the function that Ray Tune will call for each hyperparameter combination.
For each trial:
1.Start a nested MLflow run (grouped under a parent run for easy tracking).
2.Update the model’s hyperparameters for this trial.
3.Prepare the pipeline using BasicModel.prepare_features().
4.Train the model on the training set.
5.Predict on the validation set.
6.Compute metrics: Accuracy, Precision, Recall.
7.Log parameters and metrics to MLflow.
8.Report metrics back to Ray Tune for optimization."""

def train_with_nested_mlflow(config, X_train_in: pd.DataFrame,
                             X_valid_in: pd.DataFrame,
                             y_train_in: pd.DataFrame,
                             y_valid_in: pd.DataFrame,
                             project_config: ProjectConfig,
                             experiment_id: str,
                             parent_run_id: str):

    n_estimators, max_depth, learning_rate = (
        config["n_estimators"],
        config["max_depth"],
        config["learning_rate"],
    )
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(
        run_name=f"trial_n{n_estimators}_md{max_depth}_lr{learning_rate}",
        nested=True,
        parent_run_id=parent_run_id,
        experiment_id=experiment_id
    ):
        # Update parameters for this trial
        trial_params = dict(project_config.parameters)
        trial_params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        })

        # Train model
        model = BasicModel(config=project_config, tags=tags, spark=None)
        model.parameters = trial_params
        model.prepare_features()
        model.pipeline.set_params(
            classifier__n_estimators=n_estimators,
            classifier__max_depth=max_depth,
            classifier__learning_rate=learning_rate
        )
        model.pipeline.fit(X_train_in, y_train_in)

        y_pred = model.pipeline.predict(X_valid_in)
        metrics = {
            "accuracy": accuracy_score(y_valid_in, y_pred),
            "precision": precision_score(y_valid_in, y_pred),
            "recall": recall_score(y_valid_in, y_pred)
        }
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        tune.report(metrics)


# COMMAND ----------

def define_by_run_func(trial):
    trial.suggest_int("n_estimators", 100, 600, log=True)
    trial.suggest_int("max_depth", 3, 15)
    trial.suggest_float("learning_rate", 0.01, 0.2)

# Define Optuna search algo
algo = OptunaSearch(space=define_by_run_func, metric="recall", mode="max")
# Note: A concurrency limiter, limits the number of parallel trials. This is important for Bayesian search (inherently sequential) as too many parallel trials reduces the benefits of priors to inform the next search round.
#algo = ConcurrencyLimiter(algo, max_concurrent=num_cpu_cores_per_worker*max_worker_nodes+num_cpus_head_node)

# COMMAND ----------

# --- Launch Ray Tune experiment with MLflow parent run ---
import ray
import os

from mlflow.utils.databricks_utils import get_databricks_env_vars

mlflow_dbrx_creds = get_databricks_env_vars("databricks")
os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds['DATABRICKS_HOST']
os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds['DATABRICKS_TOKEN']

# for distributed, use this:

# ray_conf = setup_ray_cluster(
#   min_worker_nodes=2,
#   max_worker_nodes=2,
#   num_cpus_head_node=1,
#   num_cpus_worker_node=2,
# )
# os.environ['RAY_ADDRESS'] = ray_conf[0]

n_trials = 20

mlflow.enable_system_metrics_logging()
mlflow.set_experiment("/Shared/hotel-reservation-finetuning")
experiment_id = mlflow.get_experiment_by_name("/Shared/hotel-reservation-finetuning").experiment_id
with mlflow.start_run(
    run_name=f"optuna-finetuning-{datetime.now().strftime('%Y-%m-%d')}",
    tags={"git_sha": "1234567890abcd", "branch": "main"},
    description="LightGBM hyperparameter tuning with Ray & Optuna") as parent_run:

    tuner = tune.Tuner(
        ray.tune.with_parameters(
            train_with_nested_mlflow,
            X_train_in = X_train,
            y_train_in = y_train,
            X_valid_in = X_valid,
            y_valid_in = y_valid,
            project_config=config,
            parent_run_id=parent_run.info.run_id,
            experiment_id=experiment_id,
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=n_trials,
            reuse_actors = True # Highly recommended for short training jobs (NOT RECOMMENDED FOR GPU AND LONG TRAINING JOBS)
            ),
        # run_config=train.RunConfig(
        #     name="ray-tune-optuna",
        #     callbacks=[
        #         MLflowLoggerCallback(
        #             experiment_name=experiment_name,
        #             save_artifact=False,
        #             tags={"mlflow.parentRunId": parent_run.info.run_id})]
        # )
    )
    results = tuner.fit()

# --- Retrieve best parameters ---
best_result = results.get_best_result(metric="recall", mode="max")
print("Best hyperparameters:", best_result.config)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Reference docs:
# MAGIC - https://docs.ray.io/en/latest/index.html$0
# MAGIC - https://github.com/databricks-industry-solutions/ray-framework-on-databricks/blob/main/Hyperparam_Optimization/1-HPO-ML-Training-Optuna/01_hpo_optuna_ray_train.py#L502$0
# MAGIC - https://docs.databricks.com/aws/en/machine-learning/ray/ray-create$0
# MAGIC - https://docs.ray.io/en/latest/tune/key-concepts.html$0
# MAGIC
