from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from noise import injectGaussianNoise

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
chosenRS = 400

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= chosenRS, test_size=0.25)

# Features identified as important/unimportant (manually or via SHAP/feature importance)
# Get the top 3 most and least important features
# important_features = feature_importance_df.nlargest(3).index.tolist() # ['perimeter3', 'concave_points1', 'radius3']
# unimportant_features = feature_importance_df.nsmallest(3).index.tolist() # ['smoothness2', 'fractal_dimension1', 'symmetry2']
important_features = ["perimeter3", "concave_points1", "radius3"]

# Noise levels to test
noise_levels = [0, 0.1, 0.5, 1, 2]  # Standard deviations

# Models to compare
models = {
    "RandomForest": RandomForestClassifier(criterion="gini", random_state= chosenRS),
    "DecisionTree": DecisionTreeClassifier(random_state= chosenRS),
    "NeuralNetwork": MLPClassifier(random_state= chosenRS, max_iter=1000),
}

# DataFrame to store results
results = []

for model_name, model in models.items():
    for feature_set, feature_list in [("Important", important_features), ("Unimportant", unimportant_features)]:
        for noise_std in noise_levels:
            # Create a copy of the dataset
            X_train_noisy = X_train.copy()
            X_test_noisy = X_test.copy()
            
            for feature in important_features:
                injectGaussianNoise(X_train_noisy, [feature], 0, noise_std * X_train[feature].std())

            # Train model
            model.fit(X_train_noisy, y_train.values.ravel())

            # Make predictions
            y_pred = model.predict(X_test_noisy)

            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            results.append([model_name, feature_set, noise_std, accuracy])

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Feature Set", "Noise Std", "Accuracy"])
results_df.to_csv('file1.csv')
