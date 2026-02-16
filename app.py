import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🏭 Industrial Predictive Maintenance Dashboard")

# ✅ Safe model loading (important for Render)
model_path = os.path.join("models", "model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Make sure model.pkl exists inside models folder.")
    st.stop()

model = joblib.load(model_path)

# 🔹 Input Fields
temperature = st.slider("Temperature", 20.0, 150.0, 60.0)
vibration = st.slider("Vibration", 0.0, 100.0, 40.0)
pressure = st.slider("Pressure", 0.0, 200.0, 100.0)
humidity = st.slider("Humidity", 0.0, 100.0, 50.0)
runtime_hours = st.number_input("Runtime Hours", 0, 10000, 2000)
load_percentage = st.slider("Load %", 0.0, 100.0, 70.0)
maintenance_history = st.number_input("Maintenance Count", 0, 20, 3)
plant_location = st.selectbox("Plant Location", [0, 1, 2])

# 🔹 Feature Engineering
stress_index = (
    temperature * 0.3 +
    vibration * 0.3 +
    pressure * 0.2 +
    load_percentage * 0.2
)

# 🔹 Create DataFrame
input_df = pd.DataFrame({
    "plant_location": [plant_location],
    "temperature": [temperature],
    "vibration": [vibration],
    "pressure": [pressure],
    "humidity": [humidity],
    "runtime_hours": [runtime_hours],
    "load_percentage": [load_percentage],
    "maintenance_history": [maintenance_history],
    "stress_index": [stress_index]
})

# 🔹 Prediction
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Failure (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Machine Safe (Probability: {probability:.2f})")
