import numpy as np
import pandas as pd

# parameters
num_samples = 1000
maj_var = 0.5        # variance for the majority (label 0)
label_ratio = 3      # number of majority labels per minority label
d_means = maj_var / 2
var_ratio = 2        # variance multiplier for minority labels

# Total samples: majority + minority
total_majority = num_samples * label_ratio  # for label 0
total_minority = num_samples                # for label 1

# Create the label array:
# Label 0 for the majority samples, label 1 for the minority samples.
labels = np.concatenate([np.zeros(total_majority), np.ones(total_minority)])

# Feature 1 is just equal to the label.
feature_1 = labels.copy()

# Feature 2: 
# For label 0: Gaussian noise with mean 0 and standard deviation maj_var.
# For label 1: Gaussian noise with mean d_means and standard deviation maj_var * var_ratio.
feature_2_majority = np.random.normal(loc=0, scale=maj_var, size=total_majority)
feature_2_minority = np.random.normal(loc=d_means, scale=maj_var * var_ratio, size=total_minority)
feature_2 = np.concatenate([feature_2_majority, feature_2_minority])

# Build the DataFrame
df = pd.DataFrame({
    "feature 1": feature_1,
    "feature 2": feature_2,
    "label": labels
})

# Quick preview of the DataFrame
print(df.head())
print(df['label'].value_counts())

y = df['label']
X = df.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(X_subset, y, random_state=GLOBAL_RANDOM_STATE, test_size=0.25)


model = RandomForestClassifier(n_estimators = 10, max_depth = 15, criterion="entropy", random_state=GLOBAL_RANDOM_STATE)
model.fit(X_train, y_train.values.ravel())


