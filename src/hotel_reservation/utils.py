"""Utility class."""

import os

import numpy as np


def adjust_predictions(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Adjust classification predictions using a custom threshold.

    :param predictions: Array of predicted probabilities (for the positive class in binary classification)
    :param threshold: Decision threshold to convert probabilities into class labels
    :return: Adjusted predictions array (0 or 1)
    """
    return np.array([1 if prob >= threshold else 0 for prob in predictions])


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment.

    :return: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ
