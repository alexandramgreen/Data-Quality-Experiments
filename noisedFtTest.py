import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from noise import injectGaussianNoise

GLOBAL_RANDOM_STATE = 10249283
np.random.seed(GLOBAL_RANDOM_STATE)

path = r".\uciml\covtype.csv"
df = pd.read_csv(path)
df = df.drop(df.filter(like="Soil_Type").columns, axis=1)
df = df.drop(df.filter(like="Wilderness_Area").columns, axis=1)
df = df.sample(n=5000, random_state=GLOBAL_RANDOM_STATE)

y = df['Cover_Type']
X = df.drop(columns=['Cover_Type'])
X['Noise'] = np.random.normal(0, 1, 5000)

# Define parameters
num_trials = 5  # Number of trials per batch size
CHOICE_RANDOMS = np.random.randint(0, 999999, num_trials)

for trial in range(num_trials):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=GLOBAL_RANDOM_STATE, test_size=0.25)

    # Train baseline model (no noise)
    model = RandomForestClassifier(criterion="gini", random_state=GLOBAL_RANDOM_STATE)
    model.fit(X_train, y_train.values.ravel())

    # Get feature importances (before noising)
    importance_features = {X_train.columns[i]: float(model.feature_importances_[i]) for i in range(len(X_train.columns))}
    print(importance_features)

    # Get baseline accuracy
    y_pred = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    # Apply noise to each feature in the batch separately
    for feature in X_train.columns: 
        # Create copies of X_train and X_test
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()            
        # Inject noise into the selected feature
        injectGaussianNoise(X_train_noisy, [feature], 0, 1, GLOBAL_RANDOM_STATE)
            
        # Train model on noisy data
        model.fit(X_train_noisy, y_train.values.ravel())
        y_pred = model.predict(X_test_noisy)
        trial_accuracy = accuracy_score(y_test, y_pred)

        curr_importance_features = {X_train_noisy.columns[i]: float(model.feature_importances_[i]) for i in range(len(X_train_noisy.columns))}
                
        prev_importance = importance_features[feature]
        new_importance = curr_importance_features[feature]

