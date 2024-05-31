# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:03:45 2024

@author: niru
"""

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

model_df = {}

def model_val(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(Y_test, Y_pred)}")
    score = cross_val_score(model, X, Y, cv=5)
    print(f"{model} Avg cross val score is.{np.mean(score)}")
    model_df[model] = round(np.mean(score) * 100, 2)

def train_models(X, Y):
    models = [
        LogisticRegression(),
        svm.SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    for model in models:
        model_val(model, X, Y)

    print(model_df)
    return model_df
