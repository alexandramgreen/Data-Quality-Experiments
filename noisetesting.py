from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
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
X['constant'] = 0
X['random'] = np.random.normal(0, 1, X['constant'].shape)
print(X['random'])



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100001, test_size=0.25)
model =  RandomForestClassifier(criterion="gini", random_state= 100001)
model.fit(X_train, y_train.values.ravel())

importanceFeatures = {X.columns[i]: float(model.feature_importances_[i]) for i in range(len(X.columns))}
print(importanceFeatures['constant'])
print(importanceFeatures['random'])
print(importanceFeatures)