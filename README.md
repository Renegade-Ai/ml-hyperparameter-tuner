# ML Hyperparameter Tuner 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade Python library for automated hyperparameter tuning of machine learning models. Designed for both beginners and experts who want efficient, reliable model optimization without the complexity.

## ✨ Features

- 🎯 **Simple Interface**: One-line hyperparameter tuning for popular ML models
- 🧠 **Smart Defaults**: Pre-configured parameter spaces based on ML best practices
- 📊 **Built-in Evaluation**: Comprehensive model comparison and performance tracking
- ⚡ **Efficient Search**: Grid search and random search with parallel processing
- 🔍 **Overfitting Detection**: Automatic train/validation performance comparison
- 📈 **Feature Importance**: Built-in analysis for tree-based models
- 🔄 **Reproducible**: Consistent results with proper random state handling

## 🚀 Quick Start

```python
from hypertuner import ModelTuner
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tuner
tuner = ModelTuner('random_forest', task_type='classification')

# Quick baseline with defaults
tuner.fit_default(X_train, y_train)
tuner.evaluate(X_test, y_test)

# Automated hyperparameter tuning
tuner.tune_hyperparameters(X_train, y_train, method='random', n_iter=50)
tuner.compare_models(X_train, y_train, X_test, y_test)

# Get optimized model for production
best_model = tuner.get_model()
```
