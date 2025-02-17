from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from noise import injectGaussianNoise

# 569 observations, 30 features
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

# metadata 
# print(breast_cancer_wisconsin_diagnostic.metadata) 
# variable information 
# print(breast_cancer_wisconsin_diagnostic.variables) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 205, test_size = 0.25)

clf = RandomForestClassifier(criterion = "gini", random_state = 205)
clf.fit(X_train, y_train)
# clf = MLPClassifier(random_state = 80, max_iter = 1000)

# Get the top 3 most and least important features
# important_features = feature_importance_df.nlargest(3).index.tolist() # ['perimeter3', 'concave_points1', 'radius3']
# unimportant_features = feature_importance_df.nsmallest(3).index.tolist() # ['smoothness2', 'fractal_dimension1', 'symmetry2']


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)
    print(classification_report(y_test, y_pred))

y_pred = prediction(X_test, clf)
cal_accuracy(y_test, y_pred)

'''
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=list(X.columns), class_names=(list(set(y.values.flatten()))), filled=True)
plt.show()
'''






