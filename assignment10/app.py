import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Predict heart disease using the AdaBoost Boosting model.")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "heart_disease_boosting.pkl"
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
)
cp = st.number_input(
    "Chest Pain Type (cp)",
    min_value=0,
    max_value=3,
    value=1,
    step=1
)
chol = st.number_input(
    "Cholesterol (chol)",
    min_value=50,
    max_value=700,
    value=240
)
thalach = st.number_input(
    "Maximum Heart Rate (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

if st.button("Predict", use_container_width=True):
    input_data = pd.DataFrame(
        [[age, sex, cp, chol, thalach]],
        columns=["age", "sex", "cp", "chol", "thalach"]
    )

    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("Heart disease detected.")
        else:
            st.success("No heart disease detected.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
