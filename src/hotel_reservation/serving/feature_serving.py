"""Feature Serving module."""

from databricks import feature_engineering
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class FeatureServing:
    """Manages feature serving operations in Databricks."""

    def __init__(self, feature_table_name: str, feature_spec_name: str, endpoint_name: str) -> None:
        """Initialize the FeatureServing instance.

        :param feature_table_name: Name of the feature table
        :param feature_spec_name: Name of the feature specification
        :param endpoint_name: Name of the serving endpoint
        """
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient()
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.fe = feature_engineering.FeatureEngineeringClient()

    def create_or_update_online_table(self, online_store_name: str) -> None:
        """Create or update an online table for hotel reservation predictions."""
        fe = FeatureEngineeringClient()
        online_store = fe.get_online_store(name=online_store_name)

        # Publish the feature table to the online store
        fe.publish_table(
            online_store=online_store,
            source_table_name=self.feature_table_name,
            online_table_name=f"{self.feature_table_name}_online",
        )

    def create_feature_spec(self) -> None:
        """Create a feature spec to enable feature serving."""
        # A configuration for which features to serve.
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="Booking_ID",
                feature_names=["no_of_weekend_nights", "no_of_week_nights"],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)

    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small", scale_to_zero: bool = True) -> None:
        """Deploy or update the feature serving endpoint in Databricks.

        :param workload_size: Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: If True, endpoint scales to 0 when unused
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        served_entities = served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name, scale_to_zero_enabled=scale_to_zero, workload_size=workload_size
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
