
import streamlit as st
from irrigation_model import train_model, predict_irrigation

st.set_page_config(
    page_title="Smart Irrigation AI",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return train_model()

st.title("AI-Based Smart Irrigation System")
st.write(
    "Estimate crop irrigation needs using "
    "soil and weather conditions."
)

st.info(
    "Educational prototype using sample data. "
    "It is not connected to real sensors or a pump."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

st.header("Enter Conditions")

soil_moisture = st.slider(
    "Soil Moisture (%)", 0, 100, 30
)

temperature = st.slider(
    "Temperature (°C)", 0, 50, 30
)

humidity = st.slider(
    "Air Humidity (%)", 0, 100, 50
)

rainfall_mm = st.number_input(
    "Recent Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=0.0,
    step=1.0
)

crop_type = st.selectbox(
    "Select Crop",
    ["Rice", "Wheat", "Maize", "Tomato"]
)

if st.button("Predict Irrigation Need"):
    prediction, confidence = predict_irrigation(
        model,
        soil_moisture,
        temperature,
        humidity,
        rainfall_mm,
        crop_type
    )

    st.subheader("Prediction")

    if prediction == "Yes":
        st.warning(
            "The model predicts that irrigation may be needed."
        )
    else:
        st.success(
            "The model predicts that irrigation may not be needed."
        )

    st.write(f"Model confidence: {confidence:.1%}")

    st.caption(
        "This prediction is based on sample training data "
        "and is not a guarantee of actual crop requirements."
    )

st.divider()
st.caption("Student Project | Smart Agriculture")
