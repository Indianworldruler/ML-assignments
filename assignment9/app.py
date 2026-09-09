import streamlit as st
import pandas as pd
import pickle

st.title("Heart Disease Prediction")
st.write("Enter the patient details and click Predict.")

age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.selectbox("Sex", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
thalach = st.sidebar.number_input("Maximum Heart Rate", 60, 220, 150)

input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "chol": chol,
    "thalach": thalach
}])

if st.button("Predict"):
    with open("heart_disease_ensemble.pkl", "rb") as file:
        model = pickle.load(file)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Heart disease detected")
    else:
        st.success("No heart disease detected")
