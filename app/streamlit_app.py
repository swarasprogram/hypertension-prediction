import streamlit as st
import joblib
import pandas as pd

import urllib.request
import os

model_path = "/Users/swaradajoshi/Documents/hypertension-prediction/models/hypertension_model.joblib"
if not os.path.exists(model_path):
    # fallback if someone runs from a different directory
    model_path = "models/hypertension_model.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please make sure you have trained and saved the model.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.title("Hypertension Stage Prediction System")

age = st.slider("Age",18,80,30)
height = st.slider("Height (cm)",140,200,170)
weight = st.slider("Weight (kg)",40,120,70)

cholesterol = st.selectbox("Cholesterol Level",[1,2,3])
glucose = st.selectbox("Glucose Level",[1,2,3])

smoke = st.selectbox("Smoking",[0,1])
alco = st.selectbox("Alcohol",[0,1])
active = st.selectbox("Physically Active",[0,1])

bmi = weight / ((height/100)**2)

input_data = pd.DataFrame({
    "gender":[1],
    "height":[height],
    "weight":[weight],
    "cholesterol":[cholesterol],
    "gluc":[glucose],
    "smoke":[smoke],
    "alco":[alco],
    "active":[active],
    "age_years":[age],
    "bmi":[bmi]
})

if st.button("Predict Hypertension Stage"):
    try:
        prediction = model.predict(input_data)[0]
        st.subheader(f"Predicted Stage: {prediction}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")