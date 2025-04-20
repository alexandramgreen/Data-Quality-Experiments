import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from noise import injectGaussianNoise

GLOBAL_RANDOM_STATE = 9129384
np.random.seed(GLOBAL_RANDOM_STATE)


path = r".\uciml\covtype.csv"
df = pd.read_csv(path)
df = df.drop(df.filter(like="Soil_Type").columns, axis=1)
df = df.drop(df.filter(like="Wilderness_Area").columns, axis=1)
df = df.sample(n=5000, random_state= GLOBAL_RANDOM_STATE)

y = df['Cover_Type']
X = df.drop(columns=['Cover_Type'])

for i in range(1, 8):
    print(i)
    print(len(df[df['Cover_Type'] == i]))
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = GLOBAL_RANDOM_STATE, test_size=0.25, shuffle = True)

# injectGaussianNoise(X_train, [col for col in X_train.columns], 0, 0, GLOBAL_RANDOM_STATE)

'''
model = RandomForestClassifier(criterion="gini", random_state= GLOBAL_RANDOM_STATE)
model.fit(X_train, y_train.values.ravel())

importanceFeatures = {X_train.columns[i]: float(model.feature_importances_[i]) for i in range(len(X_train.columns))}
y_pred = model.predict(X_test)
print("Baseline: " + str(accuracy_score(y_test, y_pred)))


stateList = np.random.randint(1, 100, size = 5)
noise_levels = [0.125 * x for x in range(0, 21)]  # Standard deviations

# DataFrame to store results
results = []
for feature in ['Elevation']:
    for noise_std in noise_levels:
        sum_accuracy = 0
        for rst in stateList:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rst, test_size=0.25)
            model =  RandomForestClassifier(criterion="gini", random_state= rst)

            X_train_noisy = X_train.copy()
            X_test_noisy = X_test.copy()
                
            injectGaussianNoise(X_train_noisy, [feature], 0, noise_std, rst)

            model.fit(X_train_noisy, y_train.values.ravel())

            y_pred = model.predict(X_test_noisy)

            # Store results
            sum_accuracy += accuracy_score(y_test, y_pred)
        results.append([feature, noise_std, sum_accuracy/(len(stateList)), importanceFeatures[feature]])

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=["Feature", "Noise Std Dev", "Accuracy", "Feature Importance"])
results_df.to_csv('forestCoverTrial1.csv')
'''
