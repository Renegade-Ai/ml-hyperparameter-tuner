import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from hypertuner import ModelTuner

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model tuner
tuner = ModelTuner(
    model_name="random_forest",  # You can choose: random_forest, gradient_boosting, logistic_regression, svm, neural_network
    task_type="classification",  # or "regression"
    random_state=42,
)

# Fit with default parameters
tuner.fit_default(X_train, y_train)

# Evaluate default model
print("\nDefault Model Performance:")
tuner.evaluate(X_test, y_test, X_train, y_train)

# Tune hyperparameters
tuner.tune_hyperparameters(
    X_train, y_train, method="grid", cv=5, verbose=True  # or "random"
)

# Evaluate tuned model
print("\nTuned Model Performance:")
tuner.evaluate(X_test, y_test, X_train, y_train)

# Compare models
tuner.compare_models(X_train, y_train, X_test, y_test)

# Get feature importance
print("\nFeature Importance:")
feature_names = [f"Feature_{i}" for i in range(20)]
tuner.get_feature_importance(feature_names)
