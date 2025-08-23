import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from hypertuner import ModelTuner

# Create a more challenging synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,  # Most features are informative
    n_redundant=3,     # Some redundant features
    n_repeated=2,      # Some repeated features
    n_classes=2,
    random_state=42,
    class_sep=1.0,     # Reasonable class separation
    flip_y=0.1         # Add some label noise
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# List of all classification models available
models = [
    "random_forest",
    "gradient_boosting",
    "logistic_regression",
    "svm",
    "neural_network"
]

# Dictionary to store results
results = {
    'Model': [],
    'Default Accuracy': [],
    'Tuned Accuracy': [],
    'Improvement': [],
    'Best Parameters': []
}

# Test each model
for model_name in models:
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    # Initialize model
    tuner = ModelTuner(
        model_name=model_name,
        task_type="classification",
        random_state=42
    )
    
    # Train with default parameters
    tuner.fit_default(X_train, y_train)
    default_score = tuner.evaluate(X_test, y_test)
    
    # Tune hyperparameters
    tuner.tune_hyperparameters(
        X_train,
        y_train,
        method="grid",  # Use grid search for thorough comparison
        cv=5,
        verbose=True
    )
    
    # Evaluate tuned model
    tuned_score = tuner.evaluate(X_test, y_test)
    
    # Store results
    results['Model'].append(model_name)
    results['Default Accuracy'].append(default_score)
    results['Tuned Accuracy'].append(tuned_score)
    results['Improvement'].append(tuned_score - default_score)
    results['Best Parameters'].append(tuner.tuning_param['best_params'])
    
    # Compare models
    tuner.compare_models(X_train, y_train, X_test, y_test)

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Tuned Accuracy', ascending=False)

print("\n\nFinal Results Summary:")
print("=" * 80)
print(results_df[['Model', 'Default Accuracy', 'Tuned Accuracy', 'Improvement']].to_string(index=False))

# Save detailed results
results_df.to_csv('model_comparison_results.csv', index=False)
print("\nDetailed results saved to 'model_comparison_results.csv'")
