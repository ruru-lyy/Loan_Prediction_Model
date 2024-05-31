# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:05:57 2024

@author: niru
"""

from tkinter import *
import joblib
import pandas as pd

def show_entry():
    try:
        p1 = float(e1.get())
        p2 = float(e2.get())
        p3 = float(e3.get())
        p4 = float(e4.get())
        p5 = float(e5.get())
        p6 = float(e6.get())
        p7 = float(e7.get())
        p8 = float(e8.get())
        p9 = float(e9.get())
        p10 = float(e10.get())
        p11 = float(e11.get())
        
        model = joblib.load('loan_status_predict.joblib')
        
        df = pd.DataFrame({
            'Gender': [p1],
            'Married': [p2],
            'Dependents': [p3],
            'Education': [p4],
            'Self_Employed': [p5],
            'ApplicantIncome': [p6],
            'CoapplicantIncome': [p7],
            'LoanAmount': [p8],
            'Loan_Amount_Term': [p9],
            'Credit_History': [p10],
            'Property_Area': [p11]
        })
        
        result = model.predict(df)
        
        if result[0] == 1:
            result_text.set("Loan Approved")
        else:
            result_text.set("Loan Not Approved")
    except Exception as e:
        result_text.set(f"Error: {str(e)}")

master = Tk()
master.title("Loan Status Prediction Using Machine Learning")

label = Label(master, text="Loan Status Prediction", bg="black", fg="white")
label.grid(row=0, columnspan=2)

Label(master, text="Gender [1:Male ,0:Female]").grid(row=1)
Label(master, text="Married [1:Yes,0:No]").grid(row=2)
Label(master, text="Dependents [1,2,3,4]").grid(row=3)
Label(master, text="Education [0:Not Graduate, 1:Graduate]").grid(row=4)
Label(master, text="Self_Employed [0:No, 1:Yes]").grid(row=5)
Label(master, text="ApplicantIncome").grid(row=6)
Label(master, text="CoapplicantIncome").grid(row=7)
Label(master, text="LoanAmount").grid(row=8)
Label(master, text="Loan_Amount_Term").grid(row=9)
Label(master, text="Credit_History [0:No, 1:Yes]").grid(row=10)
Label(master, text="Property_Area [0:Urban, 1:Semiurban, 2:Rural]").grid(row=11)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)

Button(master, text="Predict", command=show_entry).grid(row=12, columnspan=2)

result_text = StringVar()
Label(master, textvariable=result_text).grid(row=13, columnspan=2)

mainloop()
