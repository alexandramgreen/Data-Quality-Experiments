import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from noise import injectGaussianNoise

# Set global random seed
GLOBAL_RANDOM_STATE = 91137
np.random.seed(GLOBAL_RANDOM_STATE)

# Load dataset
path = r".\uciml\covtype.csv"
df = pd.read_csv(path)

# Drop categorical variables
df = df.drop(df.filter(like="Soil_Type").columns, axis=1)
df = df.drop(df.filter(like="Wilderness_Area").columns, axis=1)

# Sample dataset for efficiency
# df = df.sample(n=10000, random_state=GLOBAL_RANDOM_STATE)

# Convert Cover_Type into a binary classification problem (2 vs. all other types)
df['Cover_Type'] = (df['Cover_Type'] == 2).astype(int)  # 1 if Cover_Type == 2, otherwise 0

# Define predictor variables and target
y = df['Cover_Type']
X = df.drop(columns=['Cover_Type'])

# Define parameters
batch_sizes = [3, 5, 9]  # Different batch sizes
num_trials = 15  # Number of trials per batch size
noise_levels = [0.5, 1.0, 2.0]  # Standard deviations

# DataFrame to store results
results = []

# Loop over different batch sizes
for batch_size in batch_sizes:
    if batch_size == 9:
        num_trials = 1
        
    for trial in range(num_trials):
        
        # Select a random batch of features
        sampled_features = np.random.choice(X.columns, batch_size, replace=False)
        X_subset = X[sampled_features]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, random_state=GLOBAL_RANDOM_STATE, test_size=0.25)

        # Train baseline model (no noise) with Logistic Regression
        model = LogisticRegression(random_state=GLOBAL_RANDOM_STATE, max_iter=500, solver='liblinear')
        model.fit(X_train, y_train.values.ravel())

        # Get feature coefficients (importance scores before noising)
        importance_features = {sampled_features[i]: abs(float(model.coef_[0][i])) for i in range(len(sampled_features))}

        importance_sum = sum(importance_features.values())

        # Get baseline accuracy
        y_pred = model.predict(X_test)
        baseline_accuracy = precision_score(y_test, y_pred, pos_label=1)
        
        # Store baseline result
        results.append([trial, batch_size, "Baseline", baseline_accuracy, 0, 0, 0, 1])

        # Apply noise to each feature in the batch separately
        for feature in sampled_features:
            for noise_std in noise_levels:
                
                # Create copies of X_train and X_test
                X_train_noisy = X_train.copy()
                X_test_noisy = X_test.copy()
                
                # Inject noise into the selected feature
                injectGaussianNoise(X_train_noisy, [feature], 0, noise_std, GLOBAL_RANDOM_STATE)
                
                # Train model on noisy data
                model.fit(X_train_noisy, y_train.values.ravel())
                y_pred = model.predict(X_test_noisy)
                trial_accuracy = precision_score(y_test, y_pred, pos_label=1)
                absolute_importance = importance_features[feature] * importance_sum
                # Store results
                results.append([trial, batch_size, feature, trial_accuracy, noise_std, importance_features[feature], baseline_accuracy - trial_accuracy, trial_accuracy/baseline_accuracy])

# Convert results to a DataFrame and save
results_df = pd.DataFrame(results, columns=["Trial Number", "Batch Size", "Feature Noised", "Precision", "Noise Std Dev", "Feature Importance", "Degradation in Precision Difference", "Degradation in Precision Ratio"])
results_df.to_csv('forestCover_trials_logistic_CoverType2.csv', index=False)
