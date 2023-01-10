# -*- coding: utf-8 -*-
"""
Spyder Editor

LUNG CANCER DATASET.

OBTAINED FROM KAGGLE:
https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('survey lung cancer.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# This dataset has a lot of binary characteristics classified in 1 (NO) - 2 (YES) 

# =============================================================================
# For this dataset is important to preprocess two columns of data which
# are the Age one and the last one that indicates if the patient has lung cancer or not
# =============================================================================
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
labelencoder_y = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(np.array(x[:, 0]))
y = labelencoder_y.fit_transform(y)



# =============================================================================
# Variable scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
# scaler_y= StandardScaler()
x = scaler_x.fit_transform(x)
# y = scaler_y.fit_transform(y)


# =============================================================================
# Train - test split
# =============================================================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)


# =============================================================================
# Creation of different Classification Models
# =============================================================================

# MULTIPLE REGRESSION
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()
log_regression.fit(x_train, y_train)

py = log_regression.predict(x_test)

# SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

py = svc.predict(x_test)


# NAIVE-BAYES
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)

py = naive_bayes.predict(x_test)


# K NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

py = knn.predict(x_test)


# DECISSION TREES
from sklearn.tree import DecisionTreeClassifier
decission_tree = DecisionTreeClassifier()
decission_tree.fit(x_train, y_train)

py = decission_tree.predict(x_test)


# =============================================================================
# Confussion matrix to confirm the accuracy of the model
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, py)

