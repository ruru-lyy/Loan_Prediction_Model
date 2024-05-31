# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:03:05 2024

@author: niru
"""

import pandas as pd

def load_and_preprocess_data(filepath):
    pd.set_option('display.max_columns', None)
    data = pd.read_csv(filepath)
    print(data.head())
    print('Dimensions of our data')
    print(data.shape)
    print(data.info())

    # Handling missing values
    print(data.isnull().sum() * 100 / len(data))
    columns = ['Gender', 'Married', 'Dependents', 'LoanAmount', 'Loan_Amount_Term']
    data = data.dropna(subset=columns)
    data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
    data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
    print(data.isnull().sum() * 100 / len(data))

    # Handling Categorical Data
    data['Dependents'] = data['Dependents'].replace(to_replace="3+", value='4')
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}).astype('int')
    data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
    data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
    
    # Storing Feature Matrix in X and Target in vector Y
    X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
    Y = data['Loan_Status']

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    scaler = StandardScaler()
    X[cols] = scaler.fit_transform(X[cols])
    
    return X, Y
