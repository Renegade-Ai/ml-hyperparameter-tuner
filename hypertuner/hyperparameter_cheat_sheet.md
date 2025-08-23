# Hyperparameter Tuning Cheat Sheet

This document summarizes the key hyperparameters used in pipeline across different models, what they control, their effects, and common production practices.

---

## 🔹 TF-IDF Parameters (common to all)

- **`tfidf__max_features`**

  - Limits vocabulary size to top-N terms.
  - Small → faster, less expressive (risk underfitting).
  - Large → richer, slower (risk overfitting).
  - **Prod**: Adjust based on dataset size (500–5000 typical).

- **`tfidf__ngram_range`**
  - (1,1) = unigrams; (1,2) = unigrams + bigrams.
  - Adding bigrams captures context (“machine learning” vs “machine” + “learning”).
  - **Prod**: (1,2) common; (1,3) increases feature space a lot.

---

## 🔹 Random Forest

- **`n_estimators`**
  - Number of trees.
  - ↑ → more stable, slower.
  - **Prod**: 100–500 typical.
- **`max_depth`**
  - Depth of trees.
  - ↑ → complex, risk overfit; ↓ → simpler, more bias.
  - **Prod**: None (big data), else 10–30.
- **`min_samples_split`**
  - Min samples to split a node.
  - ↑ → smoother, less overfit.
  - **Prod**: 2 or 5.
- **`min_samples_leaf`**
  - Min samples per leaf.
  - ↑ → smoother predictions.
  - **Prod**: 1 default; 2–10 if noisy.
- **`max_features`**
  - Features considered per split.
  - `sqrt` (classification default), `log2` for speed.
  - ↑ → less randomness, ↓ → better generalization.

---

## 🔹 Gradient Boosting

- **`n_estimators`**
  - Number of boosting stages.
  - ↑ → stronger model, slower.
  - **Prod**: 100–500 typical.
- **`learning_rate`**
  - Step size for updates.
  - ↓ → stable, needs more trees. ↑ → faster, risk overfit.
  - **Prod**: 0.05–0.1 typical.
- **`max_depth`**
  - Depth of base learners.
  - Shallow = better generalization.
  - **Prod**: 3–6.
- **`subsample`**
  - Fraction of samples per tree.
  - <1.0 → randomness, less variance.
  - **Prod**: 0.8–0.9 common.
- **`min_samples_split`**
  - Same as RF, affects weak learners.
  - ↑ → smoother splits.

---

## 🔹 Logistic Regression

- **`penalty`** (l1, l2, elasticnet)
  - l1 = sparsity; l2 = shrinkage; elasticnet = mix.
  - **Prod**: l2 default, l1 for feature selection.
- **`C`**
  - Inverse reg. strength.
  - ↓ → stronger reg, ↑ → weaker reg.
  - **Prod**: 0.01–10.
- **`solver`**
  - Optimizer choice.
  - liblinear = small data; saga = scalable, supports elasticnet.
  - **Prod**: saga for large text datasets.

---

## 🔹 Support Vector Machine (SVM)

- **`C`**
  - Regularization parameter.
  - ↓ → wider margin, more bias. ↑ → narrower, overfit risk.
  - **Prod**: 0.01–10 tuned.
- **`kernel`**
  - Feature mapping.
  - linear = text data (scalable). rbf/poly = complex, slower.
  - **Prod**: linear for NLP.
- **`gamma`**
  - For rbf/poly only.
  - ↓ → smoother; ↑ → overfit risk.
  - **Prod**: "scale" default.

---

## 🔹 Neural Network (MLPClassifier/Regressor)

- **`hidden_layer_sizes`**
  - Shape of hidden layers (e.g., (100,), (50,50)).
  - ↑ layers/units → more capacity, risk overfit.
  - **Prod**: Start small, grow if needed.
- **`learning_rate_init`**
  - Initial step size.
  - ↓ → stable but slow. ↑ → fast but unstable.
  - **Prod**: 0.001–0.01.
- **`alpha`**
  - L2 reg on weights.
  - ↑ → stronger reg, less overfit.
  - **Prod**: 0.0001 default; tune if overfitting.
- **`activation`**
  - Non-linearity (relu/tanh).
  - relu = fast, standard; tanh sometimes for small data.
  - **Prod**: relu almost always.

---

## 🔹 Linear Models (Regression)

- **Linear Regression**
  - `fit_intercept`: Learn intercept? Usually True.
  - `normalize`: Deprecated, use StandardScaler.
- **Ridge/Lasso**
  - `alpha`: Reg strength.
    - Ridge = shrinks weights. Lasso = zeros weights (feature selection).
  - **Prod**: alpha tuned via CV.

---

✅ **Production Tips**

- Tune only the most impactful parameters (others keep defaults).
- Use log scale for ranges (e.g., C, alpha, learning_rate).
- Start with RandomizedSearch for broad sweep, then GridSearch for fine tuning.
