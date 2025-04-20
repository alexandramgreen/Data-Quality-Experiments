from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from noise import injectGaussianNoise

# Global random state
GLOBAL_RANDOM_STATE = 1000

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Fix train-test split for consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=GLOBAL_RANDOM_STATE)

# Train Random Forest on ALL features
full_model = RandomForestClassifier(n_estimators=20, random_state=GLOBAL_RANDOM_STATE)
full_model.fit(X_train, y_train.values.ravel())

# Extract feature importance from the full model
feature_importances_full = {feature: importance for feature, importance in zip(X_train.columns, full_model.feature_importances_)}

# Store results
results = []

# Loop through each feature
for feature in X_train.columns:
    
    # **1️⃣ Full Model Case (All Features, Noise Only This Feature)**
    X_train_noisy = X_train.copy()
    X_test_noisy = X_test.copy()

    # Inject noise (0.5 std dev of feature)
    injectGaussianNoise(X_train_noisy, [feature], 0, 0.5 * X_train[feature].std())

    # Train and evaluate the full model with noise
    full_model.fit(X_train_noisy, y_train.values.ravel())
    y_pred_full = full_model.predict(X_test_noisy)

    # Calculate accuracy and precision
    accuracy_full = accuracy_score(y_test, y_pred_full)
    precision_full = precision_score(y_test.values.ravel(), y_pred_full, zero_division=0, pos_label="M")

    # ** Single-Feature Model Case (Only This Feature)**
    X_train_single = X_train[[feature]]
    X_test_single = X_test[[feature]]

    # Train single-feature model
    single_model = RandomForestClassifier(n_estimators=100, random_state=GLOBAL_RANDOM_STATE)
    single_model.fit(X_train_single, y_train.values.ravel())

    # Extract feature importance (will only have one value)
    feature_importance_single = single_model.feature_importances_[0]

    # Inject noise (0.5 std dev)
    X_train_single_noisy = X_train_single.copy()
    injectGaussianNoise(X_train_single_noisy, [feature], 0, 0.5 * X_train_single[feature].std())

    # Train and evaluate the single-feature model with noise
    single_model.fit(X_train_single_noisy, y_train.values.ravel())
    y_pred_single = single_model.predict(X_test_single)

    # Calculate accuracy and precision
    accuracy_single = accuracy_score(y_test, y_pred_single)
    precision_single = precision_score(y_test.values.ravel(), y_pred_single, zero_division=0, pos_label="M")

    # Append results for single-feature model
    results.append([
        feature, accuracy_single, precision_single, feature_importances_full[feature], accuracy_full, precision_full
    ])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Feature", "Singular Accuracy", 
                                            "Singular Precision", "Contextual Importance", "Contextual Accuracy", "Contextual Precision"])

# Save results to CSV
results_df.to_csv("feature_noise_experiment_all_vs_single.csv", index=False)

# Return DataFrame
print(results_df)
