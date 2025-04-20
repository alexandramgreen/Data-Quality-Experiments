import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from noise import injectGaussianNoise

# Set global random seed
GLOBAL_RANDOM_STATE = 10249283
np.random.seed(GLOBAL_RANDOM_STATE)

# Load dataset
path = r".\uciml\covtype.csv"
df = pd.read_csv(path)

# Drop categorical variables
df = df.drop(df.filter(like="Soil_Type").columns, axis=1)
df = df.drop(df.filter(like="Wilderness_Area").columns, axis=1)

# Sample dataset for efficiency
df = df.sample(n=5000, random_state=GLOBAL_RANDOM_STATE)

# Define predictor variables and target
y = df['Cover_Type']
X = df.drop(columns=['Cover_Type'])
X['noise, std dev 0.5'] = np.random.normal(0, .5, 5000)
X['noise, std dev 1'] = np.random.normal(0, 1, 5000)
X['noise, std dev 2'] = np.random.normal(0, 2, 5000)
X['constant'] = np.ones(5000)

# Define parameters
batch_sizes = [3, 5, len(X.columns)]  # Different batch sizes
num_trials = 20  # Number of trials per batch size
CHOICE_RANDOMS = np.random.randint(0, 999999, num_trials)
noise_levels = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Standard deviations

# DataFrame to store results
results = []

# Loop over different batch sizes
for batch_size in batch_sizes:
    if batch_size == len(X.columns):
        num_trials = 1
        
    for trial in range(num_trials):
        
        # Select a random batch of features
        np.random.seed(CHOICE_RANDOMS[trial])
        sampled_features = np.random.choice(X.columns, batch_size, replace=False)
        X_subset = X[sampled_features]
        np.random.seed(GLOBAL_RANDOM_STATE)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, random_state=GLOBAL_RANDOM_STATE, test_size=0.25)

        # Train baseline model (no noise)
        model = RandomForestClassifier(n_estimators = 10, max_depth = 15, criterion="entropy", random_state=GLOBAL_RANDOM_STATE)
        model.fit(X_train, y_train.values.ravel())

        # Get feature importances (before noising)
        importance_features = {sampled_features[i]: float(model.feature_importances_[i]) for i in range(len(sampled_features))}

        # Get baseline accuracy
        y_pred = model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred)
        # Store baseline result
        results.append([trial, batch_size, "Baseline", baseline_accuracy, 0, 0, 1, 0, 0, 1])

        # Apply noise to each feature in the batch separately
        for feature in sampled_features:
            for noise_std in noise_levels:
                
                # Create copies of X_train and X_test
                X_train_noisy = X_train.copy()
                X_test_noisy = X_test.copy()
                model = RandomForestClassifier(n_estimators = 10, max_depth = 15, criterion="entropy", random_state=GLOBAL_RANDOM_STATE)
                # Inject noise into the selected feature
                injectGaussianNoise(X_train_noisy, [feature], 0, noise_std, GLOBAL_RANDOM_STATE)
                if feature == 'constant':
                    X_train_noisy['constant'] = np.random.normal(0, noise_std, 3750)

                # Train model on noisy data
                model.fit(X_train_noisy, y_train.values.ravel())
                y_pred = model.predict(X_test_noisy)
                trial_accuracy = accuracy_score(y_test, y_pred)

                curr_importance_features = {sampled_features[i]: float(model.feature_importances_[i]) for i in range(len(sampled_features))}
                
                prev_importance = importance_features[feature]
                new_importance = curr_importance_features[feature]

                # Store results
                if prev_importance == 0:
                    results.append([trial, batch_size, feature, trial_accuracy, noise_std, prev_importance, "Infinity", prev_importance - new_importance, baseline_accuracy - trial_accuracy, trial_accuracy/baseline_accuracy])
                else:
                    results.append([trial, batch_size, feature, trial_accuracy, noise_std, prev_importance, new_importance/prev_importance, prev_importance - new_importance, baseline_accuracy - trial_accuracy, trial_accuracy/baseline_accuracy])

# Convert results to a DataFrame and save
results_df = pd.DataFrame(results, columns=["Trial Number", "Batch Size", "Feature Noised", "Accuracy", "Noise Std Dev", "Feature Importance", "Importance Ratio", "Importance Difference", "Degradation in Accuracy Difference", "Degradation in Accuracy Ratio"])
results_df.to_csv('forestCover_trials.csv', index=False)
