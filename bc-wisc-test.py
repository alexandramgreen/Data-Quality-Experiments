from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 80, test_size = 0.25)
# injectGaussianNoise(X_train, ["concave_points3"], 0, X_train["concave_points3"].std()**2)
# injectGaussianNoise(X_train, ["perimeter3"], 0, X_train["perimeter3"].std()**2)


# clf = DecisionTreeClassifier(criterion = "gini", random_state = 80, max_depth = 4, min_samples_leaf = 5)
clf = MLPClassifier(random_state = 80, max_iter = 1000)

# injectGaussianNoise(X_test, ["concave_points3"], 0, X_test["concave_points3"].std() ** 2)
# injectGaussianNoise(X_test, ["perimeter3"], 0, X_test["perimeter3"].std() ** 2)


#injectGaussianNoise(X_train, ["petal width"], 0, X_train["petal width"].std())

clf.fit(X_train, y_train.values.ravel())

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





