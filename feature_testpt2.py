from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import random
from noise import injectGaussianNoise

# Global random state
GLOBAL_RANDOM_STATE = 919
np.random.seed(GLOBAL_RANDOM_STATE)

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Batch sizes to test
batch_sizes = [len(X.columns)]
num_selections = 50  # Number of random feature selections per batch size

# Store results
results = []

# Fix train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=GLOBAL_RANDOM_STATE)

model = RandomForestClassifier(criterion = 'gini', random_state=GLOBAL_RANDOM_STATE)
model.fit(X_train, y_train.values.ravel())

# Compute feature importance
feature_imp_df = {'Feature': X_train.columns, 'Importance': model.feature_importances_}
importanceFeatures = {X_train.columns[i]: float(model.feature_importances_[i]) for i in range(len(X_train.columns))}

# Loop through batch sizes
for batch_size in batch_sizes:
    for _ in range(num_selections):
        # Randomly select `batch_size` features
        selected_features = random.sample(list(X.columns), batch_size)

        # Train logistic regression on the selected features
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]

        # Test noise effect on each feature
        for feature in selected_features:
            model = RandomForestClassifier(criterion = 'gini', random_state=GLOBAL_RANDOM_STATE)
            # Copy train data for noise injection
            X_train_noisy = X_train_subset.copy()
            X_test_noisy = X_test_subset.copy()

            # Inject noise (0.5 std dev of feature)
            injectGaussianNoise(X_train_noisy, [feature], 0, 0.5 * X_train_subset[feature].std())

            # Train and evaluate the model with noisy data
            model.fit(X_train_noisy, y_train.values.ravel())
            y_pred = model.predict(X_test_noisy)

            # Calculate accuracy and precision
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test.values.ravel(), y_pred, zero_division=0, pos_label="M")

            # Append results
            results.append([
                batch_size, feature, importanceFeatures[feature], accuracy, precision
            ])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Batch Size", "Feature", "Initial Importance", "Accuracy", "Precision"])

# Save results to CSV
results_df.to_csv("feature_noise_experiment.csv", index=False)


