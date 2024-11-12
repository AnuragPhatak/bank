import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib

data=pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\IMARTICUS\streamlit\bank_churn\train_bank_churn (1).xls")


model = joblib.load("bank_churn_model1.pkl")
page = st.sidebar.selectbox("Select Options",["Home","EDA","Model"])

if page == "Home":
    st.write("Home Page")
    st.title("**Bank Churn Prediction📈**")
elif page == "EDA":
    data.drop(columns=["id","Surname","CustomerId"],inplace=True)
    st.write(data.head())
    st.write(data.tail())
    rows,col = data.shape
    st.write(f"Rows: {rows} \nColumns: {col}")
    st.write("Checking Null Values")
    st.write(data.isnull().sum())
    st.write("Descripive Satistics")
    st.write(data.describe(include="all").fillna("="))

    num_col = data[["CreditScore","Age","Balance","EstimatedSalary"]]
    plt.figure(figsize=(15,9))

    for i,j in enumerate(num_col.columns,1):
        plt.subplot(2,3,i)
        plt.boxplot(num_col[j])
        plt.title(j)

    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(13,9))

    for i,j in enumerate(num_col.columns,1):
        plt.subplot(2,3,i)
        sns.histplot(num_col[j],kde=True)
        plt.title(j)

    plt.tight_layout()
    st.pyplot(plt)

    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data.Gender)

    data = pd.get_dummies(data)
    st.write(data.head())
    
else:
    st.title("Customer Churn Prediction")
    st.header("Enter Customer Details")

    # Define inputs for features
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=667)
    age = st.number_input("Age", min_value=18, max_value=100, value=33)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=145562.4, step=1000.0)

# Geography input for one-hot encoding format
    geography_france = st.checkbox("Geography: France", value=True)
    geography_germany = st.checkbox("Geography: Germany")
    geography_spain = st.checkbox("Geography: Spain")

    if st.button("Predict"):
        input_data = np.array([[credit_score,gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, 
                            estimated_salary, geography_france, geography_germany, geography_spain]])

    # Make prediction
        prediction = model.predict(input_data)

    # Display result
        if prediction[0] == 1:
            st.write("**Prediction: This customer is likely to exit.**")
        else:
            st.write("**Prediction: This customer is likely to stay.**")