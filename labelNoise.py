import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

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

# Parameters
num_trials = 5
probs_list = [0.01 * i for i in range(0, 100)]  # Probability of flipping a label
CHOICE_RANDOMS = np.random.randint(0, 2147483647, num_trials)

# Store results
results = []

# Loop over trials
for label_noise_prob in probs_list:
    sum_flips = 0
    sum_accuracy = 0
    for trial in range(num_trials):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=CHOICE_RANDOMS[trial], test_size=0.25)

        # Apply label noise to training labels
        def flip_labels(y_series, p, num_classes):
            class_set = set(y_series)
            np.random.seed(CHOICE_RANDOMS[trial])
            y_array = y_series.values.copy()
            flip_mask = np.random.rand(len(y_array)) < p
            flip_count = 0
            for i in range(len(y_array)):
                if flip_mask[i]:
                    other_classes = list(class_set - {y_array[i]})
                    y_array[i] = np.random.choice(other_classes)
                    flip_count += 1
            np.random.seed(GLOBAL_RANDOM_STATE)
            return pd.Series(y_array, index=y_series.index), flip_count

        num_classes = y.nunique()
        y_train_noised, flip_count = flip_labels(y_train, label_noise_prob, num_classes)

        # Train model on noised labels
        model = RandomForestClassifier(criterion="gini", random_state=CHOICE_RANDOMS[trial])
        model.fit(X_train, y_train_noised)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        sum_flips += flip_count
        sum_accuracy += accuracy

        # Store result
        results.append([trial, label_noise_prob, accuracy, flip_count])
    results.append(["Average", label_noise_prob, sum_accuracy/num_trials, sum_flips/num_trials])

# Save results
results_df = pd.DataFrame(results, columns=["Trial Number/Type", "Label Noise Prob", "Accuracy", "Flip Count"])
results_df.to_csv('label_noise_trials.csv', index=False)
