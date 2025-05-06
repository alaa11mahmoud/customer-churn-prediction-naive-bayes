import streamlit as pt
import joblib
import numpy as np
import pandas as pd

pt.title("📊🔍 Predict whether the customer is churn")
pt.write("This report build using Naive Bayes algorithm")



pt.sidebar.header("Input Data")
Age = pt.sidebar.slider("Age : ",5,100,30)
Tenure = pt.sidebar.slider("Tenure : ",1,200,10)
Gender = pt.sidebar.radio("Gender : ",("Female","Male"))

model = joblib.load("task_model.pkl")

g={"Female":0 , "Male":1}
x_train=np.array([[Age,Tenure,g[Gender]]])
result = model.predict(x_train)
prop = model.predict_proba(x_train)

data={1:"Churn" ,0: "Not Churn"}



pt.subheader("🔮 Prediction " )
pt.write(f"result : {data[result[0]]}")

pt.write(f"prob Not Churn : {prop[0][0]: .2%}")
pt.write(f"prob Churn : {prop[0][1] : .2%} ")
pt.bar_chart({"Not Churn":prop[0][0]*100,"Churn":prop[0][1]*100})