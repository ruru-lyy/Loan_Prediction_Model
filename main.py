# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:05:13 2024

@author: niru
"""

from data_preprocessing import load_and_preprocess_data
from model_training import train_models
from hyperparameter_tuning import tune_hyperparameters
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    filepath = "loan_prediction.csv"
    X, Y = load_and_preprocess_data(filepath)
    model_df = train_models(X, Y)
    tune_hyperparameters(X, Y)

    # Train the final model with best hyperparameters (example with RandomForest)
    rf = RandomForestClassifier(n_estimators=550, min_samples_split=100, min_samples_leaf=5,
                                max_features='sqrt', max_depth=30)
    rf.fit(X, Y)
    joblib.dump(rf, 'loan_status_predict.joblib')

if __name__ == "__main__":
    main()
