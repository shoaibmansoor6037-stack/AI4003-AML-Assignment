
import streamlit as st
import numpy as np
from pathlib import Path

# Find the saved model inside the project
MODEL_PATH = Path(__file__).resolve().parent.parent / "results" / "setB_model.npz"

model = np.load(MODEL_PATH)

theta = model["theta"]
mean = model["mean"]
std = model["std"]

st.title("Solar Power Prediction")
st.write("Predict AC power using public weather data")

hour = st.number_input(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12
)

sw_radiation = st.number_input(
    "Solar Radiation (kW/m²)",
    min_value=0.0,
    value=0.5
)

temp_2m = st.number_input(
    "Temperature at 2m (°C)",
    value=30.0
)

cloud_cover = st.number_input(
    "Cloud Cover (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

if st.button("Predict AC Power"):

    features = np.array([
        sw_radiation,
        temp_2m,
        cloud_cover
    ])

    scaled = (features - mean) / std

    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)

    X = np.array([
        1,
        scaled[0],
        scaled[1],
        scaled[2],
        sin_hour,
        cos_hour
    ])

    prediction = X @ theta

    # Power cannot be negative
    prediction = max(prediction, 0)

    st.success(
        f"Predicted AC Power: {prediction:.2f} kW"
    )
