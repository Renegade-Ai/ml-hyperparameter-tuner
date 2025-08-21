import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR


class ModelTuner:
    def __init__(
        self, model_name, task_type="classification", random_state=7, is_text_data=False
    ):
        self.model_name = model_name
        self.random_state = random_state
        self.task_type = task_type
        self.is_text_data = is_text_data

        self.model = None
        self.best_model = None
        self.tuning_results = {}
        self.is_tuned = False

        self._initialize_model()
        self._define_param_grids()

    def _initialize_model(self):
        model_map = {
            "classification": {
                "random_forest": RandomForestClassifier(random_state=self.random_state),
                "gradient_boosting": GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                "logistic_regression": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                "svm": SVC(random_state=self.random_state),
                "neural_network": MLPClassifier(
                    random_state=self.random_state, max_iter=1000
                ),
            },
            "regression": {
                "random_forest": RandomForestRegressor(random_state=self.random_state),
                "gradient_boosting": GradientBoostingRegressor(
                    random_state=self.random_state
                ),
                "linear_regression": LinearRegression(),
                "ridge": Ridge(random_state=self.random_state),
                "lasso": Lasso(random_state=self.random_state),
                "svm": SVR(),
                "neural_network": MLPRegressor(
                    random_state=self.random_state, max_iter=1000
                ),
            },
        }

        if self.task_type not in model_map:
            raise ValueError(
                f"Task type {self.task_type} is not available in our system."
            )

        if self.model_name not in model_map[self.task_type]:
            raise ValueError(
                f"Model name {self.model_name} is not available in our system."
            )

        base_model = model_map[self.task_type][self.model_name]

        # If working with text data, wrap the model in a pipeline with TF-IDF
        if self.is_text_data:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline

            self.model = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=1000,
                            stop_words="english",
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.95,
                        ),
                    ),
                    ("classifier", base_model),
                ]
            )
            print(
                f"Initialized {self.model_name} with TF-IDF preprocessing for text classification"
            )
        else:
            self.model = base_model
            print(
                f"Initialized {self.model_name} for {self.task_type} with default parameters"
            )

    def _define_param_grids(self):
        """Define hyperparameter grids for every model"""
        if self.is_text_data:
            # Parameter grids for pipeline models (with TF-IDF preprocessing)
            self.param_grids = {
                "classification": {
                    "random_forest": {
                        "classifier__n_estimators": [100, 200, 300],
                        "classifier__max_depth": [None, 10, 20, 30],
                        "classifier__min_samples_split": [2, 5, 10],
                        "classifier__min_samples_leaf": [1, 2, 4],
                        "classifier__max_features": ["sqrt", "log2"],
                        "tfidf__max_features": [500, 1000, 1500],
                        "tfidf__ngram_range": [(1, 1), (1, 2)],
                    }
                },
                "regression": {
                    "random_forest": {
                        "classifier__n_estimators": [100, 200, 300],
                        "classifier__max_depth": [None, 10, 20, 30],
                        "classifier__min_samples_split": [2, 5, 10],
                        "classifier__min_samples_leaf": [1, 2, 4],
                        "classifier__max_features": ["sqrt", "log2"],
                        "tfidf__max_features": [500, 1000, 1500],
                        "tfidf__ngram_range": [(1, 1), (1, 2)],
                    }
                },
            }
        else:
            # Parameter grids for direct models (no pipeline)
            self.param_grids = {
                "classification": {
                    "random_forest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2"],
                    }
                },
                "regression": {
                    "random_forest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2"],
                    }
                },
            }

        print("Initialized parameter grid")

    def fit_default(self, X_train, Y_train):
        """Train the model with default parameters."""
        self.model.fit(X_train, Y_train)
        return self

    def evaluate(self, X_test, Y_test, X_train=None, Y_train=None):
        if self.model is None:
            raise ValueError("Model is not initialized. Please fit the model first.")

        # Use the best model if available (after tuning), otherwise use the default model
        current_model = self.best_model if self.best_model is not None else self.model
        y_pred = current_model.predict(X_test)
        print(f"Model: {self.model_name}")
        print(f"Status: {'Tuned' if self.is_tuned else 'Default Parameters'}")
        print("-" * 40)

        if self.task_type == "classification":
            test_accuracy_score = accuracy_score(Y_test, y_pred)
            print(f"Test Accuracy: {test_accuracy_score:.3f}")
            print(f"Test Accuracy: {test_accuracy_score*100:.1f}%")

            if X_train is not None and Y_train is not None:
                train_pred = current_model.predict(X_train)
                train_accuracy_score = accuracy_score(Y_train, train_pred)
                print(f"Train Accuracy: {train_accuracy_score:.3f}")
                print(f"Train Accuracy: {train_accuracy_score*100:.1f}%")
            else:
                print("Train Accuracy: N/A")
            return test_accuracy_score

        if self.task_type == "regression":
            print("In progress")
            return None

    def tune_hyperparameters(
        self,
        X_train,
        Y_train,
        method="grid",
        cv=5,
        n_iter=10,
        n_jobs=-1,
        scoring=None,
        verbose=True,
    ):
        """Tune hyperparameters using Grid Search or Random Search."""
        if self.model is None:
            raise ValueError("Model is not initialized. Please fit the model first.")

        param_grid = self.param_grids[self.task_type][self.model_name]

        if verbose:
            print(
                f"🔧 Tuning {self.model_name} hyperparameters using {method} search..."
            )
            print(f"Search space: {len(param_grid)} parameters")
            print(param_grid.values())
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"Total combinations: {total_combinations}")

        if scoring is None:
            scoring = "accuracy" if self.task_type == "classification" else "r2"

        # Initialize search object
        if method == "grid":
            search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1 if verbose else 0,
            )
        elif method == "random":
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=1 if verbose else 0,
            )
        else:
            raise ValueError("Method must be 'grid' or 'random'")

        search.fit(X_train, Y_train)

        self.best_model = search.best_estimator_
        self.tuning_param = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_result": search.cv_results_,
        }
        self.is_tuned = True

        if verbose:
            print(f"✓ Hyperparameter tuning completed!")
            print(f"Best CV Score: {search.best_score_:.4f}")
            print(f"Best Parameters: {search.best_params_}")
        return self

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance for tree-based models.

        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        """

        current_model = self.best_model if self.best_model else self.model
        if hasattr(current_model, "feature_importances_"):
            importances = current_model.feature_importances_
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            feature_importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values(by="Importance", ascending=False)

            print(feature_importance_df.head(10))
            return feature_importance_df

    def compare_models(self, X_train, y_train, X_test, y_test):
        """
        Compare performance before and after tuning.

        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        """

        if not self.is_tuned:
            print("No tuned model available. Run tune_hyperparameters() first.")
            return

        print("\n Model Comparison:")
        print("=" * 50)

        self.model.fit(X_train, y_train)
        default_pred = self.model.predict(X_test)
        tuned_pred = self.best_model.predict(X_test)

        if self.task_type == "classification":
            default_score = accuracy_score(y_test, default_pred)
            tuned_score = accuracy_score(y_test, tuned_pred)
            metric_name = "Accuracy"
        else:
            default_score = r2_score(y_test, default_pred)
            tuned_score = r2_score(y_test, tuned_pred)
            metric_name = "R² Score"

        improvement = tuned_score - default_score
        improvement_pct = (improvement / default_score) * 100

        print(f"Default Model {metric_name}: {default_score:.4f}")
        print(f"Tuned Model {metric_name}:   {tuned_score:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")

        if improvement > 0:
            print("🎉 Hyperparameter tuning improved the model!")
        else:
            print("📝 Default parameters were already quite good.")

    def get_model(self):
        """Return the best model (tuned if available, otherwise default)."""
        return self.best_model if self.best_model is not None else self.model

    def save_model(self, filename=None):
        """Save tuning results to a file."""
        if not self.is_tuned:
            print("No tuning results to save.")
            return

        if filename is None:
            filename = f"{self.model_name}_tuning_results.txt"

        with open(filename, "w") as f:
            f.write(f"Hyperparameter Tuning Results\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Task: {self.task_type}\n")
            f.write(f"Best Score: {self.tuning_results['best_score']:.4f}\n")
            f.write(f"Best Parameters:\n")
            for param, value in self.tuning_results["best_params"].items():
                f.write(f"  {param}: {value}\n")

        print(f"Results saved to {filename}")
