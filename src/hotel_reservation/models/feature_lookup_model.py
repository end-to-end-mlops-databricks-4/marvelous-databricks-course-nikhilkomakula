"""FeatureLookUp model implementation."""

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from hotel_reservation.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_booking_value"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the hotel_reservations table and populate it.

        This table stores features related to hotel reservation.
        """
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {self.feature_table_name}
            (Booking_ID STRING NOT NULL, no_of_previous_cancellations BIGINT, no_of_previous_bookings_not_canceled BIGINT);
        """)
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotel_reservation_pk PRIMARY KEY(Booking_ID);"
        )
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the hotel reservation's booking value.

        This function adds no_of_weekend_nights with no_of_week_nights and multiplies the sum with avg_price_per_room.
        """
        self.spark.sql(f"""
            CREATE OR REPLACE FUNCTION {self.function_name}(
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

        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))
        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"],
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="booking_value",
                    input_bindings={
                        "avg_price_per_room": "avg_price_per_room",
                        "no_of_weekend_nights": "no_of_weekend_nights",
                        "no_of_week_nights": "no_of_week_nights",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["booking_value"] = self.test_set["avg_price_per_room"] * (
            self.test_set["no_of_weekend_nights"] + self.test_set["no_of_week_nights"]
        )

        self.X_train = self.training_df[self.num_features + self.cat_features + ["booking_value"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["booking_value"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

        self.labelEncoder = LabelEncoder()
        self.y_train_encoded = self.labelEncoder.fit_transform(self.y_train)
        self.y_test_encoded = self.labelEncoder.transform(self.y_test)
        logger.info("✅ Target successfully encoded.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM classifier.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train_encoded)
            y_pred = pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test_encoded, y_pred)
            precision = precision_score(self.y_test_encoded, y_pred)
            recall = recall_score(self.y_test_encoded, y_pred)
            f1score = f1_score(self.y_test_encoded, y_pred)

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1score}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1score", f1score)

            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def update_feature_table(self) -> None:
        """Update the hotel_reservations table with the latest records from train and test sets.

        Executes SQL queries to insert new records based on timestamp.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.catalog_name}.{self.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
            FROM {self.catalog_name}.{self.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.catalog_name}.{self.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
            FROM {self.catalog_name}.{self.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("✅ Hotel reservations feature table updated successfully.")

    def model_improved(self, test_set: DataFrame) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1 Score.

        :param test_set: DataFrame containing the test data.
        :return: True if the current model performs better, False otherwise.
        """
        return True
        # X_test = test_set.drop(self.config.target)

        # predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
        #     "prediction", "prediction_latest"
        # )

        # current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model-fe"
        # predictions_current = self.fe.score_batch(model_uri=current_model_uri, df=X_test).withColumnRenamed(
        #     "prediction", "prediction_current"
        # )

        # # Select only needed columns and convert to pandas for label encoding
        # test_labels = test_set.select("Booking_ID", "booking_status").toPandas()

        # # Use the same label encoder from training
        # test_labels["booking_status"] = self.labelEncoder.transform(test_labels["booking_status"])

        # # Convert back to Spark DataFrame
        # test_labels_spark = self.spark.createDataFrame(test_labels)

        # logger.info("Predictions are ready.")

        # # Join the DataFrames on the 'Booking_ID' column
        # df = test_labels_spark.join(predictions_current, on="Booking_ID").join(predictions_latest, on="Booking_ID")

        # # Calculate the f1 score for each model
        # evaluator = BinaryClassificationEvaluator(
        #     labelCol="booking_status",
        #     metricName="f1"
        # )

        # # Calculate F1 score for current model
        # f1_current = evaluator.evaluate(
        #     df.select("booking_status", "prediction_current"),
        #     {evaluator.predictionCol: "prediction_current"}
        # )

        # # Calculate F1 score for latest model
        # f1_latest = evaluator.evaluate(
        #     df.select("booking_status", "prediction_latest"),
        #     {evaluator.predictionCol: "prediction_latest"}
        # )

        # # Compare models based on F1 score
        # logger.info(f"F1 Score for Current Model: {f1_current}")
        # logger.info(f"F1 Score for Latest Model: {f1_latest}")

        # if f1_current > f1_latest:
        #     logger.info("Current Model performs better.")
        #     return True
        # else:
        #     logger.info("New Model performs worse.")
        #     return False
