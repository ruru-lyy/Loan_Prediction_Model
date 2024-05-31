# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:04:38 2024

@author: niru
"""
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def tune_hyperparameters(X, Y):
    # Logistic Regression
    log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ['liblinear']}
    rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid,
                                    n_iter=20, cv=5, verbose=True)
    rs_log_reg.fit(X, Y)
    print('Hyperparameter tuning for Logistic Regression')
    print(rs_log_reg.best_score_)
    print(rs_log_reg.best_params_)

    # SVM
    svc_grid = {'C': [0.25, 0.50, 0.75, 1], 'kernel': ['linear']}
    rs_svc = RandomizedSearchCV(svm.SVC(), param_distributions=svc_grid,
                                cv=5, n_iter=20, verbose=True)
    rs_svc.fit(X, Y)
    print('Hyperparameter tuning for SVM')
    print(rs_svc.best_score_)
    print(rs_svc.best_params_)

    # Random Forest
    rf_grid = {
        'n_estimators': np.arange(10, 1000, 10),
        'max_features': ['sqrt'],
        'max_depth': [None, 3, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20, 30, 100],
        'min_samples_leaf': [1, 2, 5, 10]
    }
    rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid,
                               cv=5, n_iter=20, verbose=1)
    rs_rf.fit(X, Y)
    print("Hyperparameter tuning for Random Forest")
    print(f"Best Score: {rs_rf.best_score_}")
    print(f"Best Parameters: {rs_rf.best_params_}")

