from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from noise import injectGaussianNoise

GLOBAL_RANDOM_STATE = 101
np.random.seed(GLOBAL_RANDOM_STATE)

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
X['constant'] = 20

'''
X = X.drop(['concavity3',
'texture1',
'radius2',
'texture3',
'perimeter2',
'symmetry3',
'compactness3',
'smoothness3',
'fractal_dimension3',
'symmetry1',
'concavity2',
'compactness2',
'smoothness2',
'texture2',
'fractal_dimension2',
'smoothness1',
'compactness1',
'symmetry2',
'fractal_dimension1',
'concave_points2',
'concavity1',
'area1',
'area2',
'radius1',
'perimeter1'], axis = 1, inplace=False)
'''


# Split dataset

# Noise levels to test
# noise_levels = [0.125 * x for x in range(0, 33)]  # Standard deviations

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = GLOBAL_RANDOM_STATE, test_size=0.25, shuffle = True)

print(X_train.describe())
injectGaussianNoise(X_train, [col for col in X_train.columns], 0, 2, GLOBAL_RANDOM_STATE)
print(X_train.describe())
# data is successfully noised

model =  RandomForestClassifier(criterion="gini", random_state= GLOBAL_RANDOM_STATE, class_weight = 'balanced')

model.fit(X_train, y_train.values.ravel())

importanceFeatures = {X_train.columns[i]: float(model.feature_importances_[i]) for i in range(len(X_train.columns))}
print(max(importanceFeatures.values()))
print(importanceFeatures)
y_pred = model.predict(X_test)
print(recall_score(y_test, y_pred, pos_label='M'))
print(precision_score(y_test, y_pred, pos_label='M'))

'''

featureList = ['concave_points3', 'perimeter3', 'area3']
stateList = [100, 200, 141, 19, 18]

# DataFrame to store results
results = []
for feature in featureList:
    for noise_std in noise_levels:
        sum_accuracy = 0
        sum_precision = 0
        for rst in stateList:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rst, test_size=0.25)
            model =  RandomForestClassifier(criterion="gini", random_state= rst)

            X_train_noisy = X_train.copy()
            X_test_noisy = X_test.copy()
                
            injectGaussianNoise(X_train_noisy, [feature], 0, noise_std * X_train[feature].std())

            model.fit(X_train_noisy, y_train.values.ravel())

            y_pred = model.predict(X_test_noisy)

            # Store results
            sum_accuracy += accuracy_score(y_test, y_pred)
            sum_precision += precision_score(y_test.values.ravel(), y_pred, zero_division=0, pos_label="M")
        results.append([feature, noise_std, sum_accuracy/(len(stateList)), sum_precision/(len(stateList)), importanceFeatures[feature]])

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=["Feature", "Noise Std Dev", "Accuracy", "Precision", "Feature Importance"])
results_df.to_csv('file1.csv')
'''