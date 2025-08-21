import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from hypertuner import ModelTuner


def test_classification_initialization():
    tuner = ModelTuner("random_forest", task_type="classification")
    assert tuner.model_name == "random_forest"
    assert tuner.task_type == "classification"
    assert tuner.model is not None


def test_regression_initialization():
    tuner = ModelTuner("linear_regression", task_type="regression")
    assert tuner.model_name == "linear_regression"
    assert tuner.task_type == "regression"


def test_fit_and_evaluate():
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    tuner = ModelTuner("random_forest", task_type="classification")
    tuner.fit_default(X, y)

    # Should be able to evaluate
    score = tuner.evaluate(X, y)
    assert 0 <= score <= 1  # Accuracy should be between 0 and 1
