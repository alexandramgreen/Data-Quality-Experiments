from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from noise import injectGaussianNoise
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features
y = iris.data.targets 
  
# metadata 
# print(iris.metadata) 
# variable information 
# print(iris.variables) 

# separating training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 50, test_size = 0.25)
# injectGaussianNoise(X_train, ["petal width", "petal length", "sepal length", "sepal width"], 0, 1)
# decision tree!
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth = 3, min_samples_leaf = 5)
clf_gini.fit(X_train, y_train)

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    # print("Predicted values:")
    # print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)
    print("Report : ",
          classification_report(y_test, y_pred))
    
y_pred = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred)

plt.figure(figsize=(12, 8))
plot_tree(clf_gini, feature_names=list(X.columns), class_names=(list(set(y.values.flatten()))), filled=True)
plt.show()