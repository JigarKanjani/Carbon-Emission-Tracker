import streamlit as st
import numpy as np
import joblib
import os

# Load the XGBoost model
@st.cache_resource
def load_model():
    model_path = "xgb_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        st.error("Model file not found. Please upload 'xgb_model.pkl'.")
        return None

model = load_model()

# Prediction function (with hardcoded test)
def predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction):
    # Hardcoded test input data to rule out issues with inputs
    input_data = np.array([[25100, 52000, 15000]])  # Test this data instead of passed inputs

    # Debugging: Inspect input before prediction
    st.write("Input data for prediction (hardcoded for test):", input_data)

    if model is not None:
        # Make prediction using the model
        prediction = model.predict(input_data)[0]

        # Debugging: Inspect raw prediction output
        st.write("Raw prediction output:", prediction)

        return prediction
    else:
        return None

# Streamlit App layout
st.title("Carbon Emissions Prediction")

st.write("""
Enter your energy consumption, GDP, and energy-GDP interaction to predict per capita carbon emissions.
""")

# Input fields for 3 features
energy_consumption = st.number_input("Per Capita Energy (kWh/year)", value=25100)
gdp_ppp = st.number_input("Per Capita GDP (PPP USD/year)", value=52000)
energy_gdp_interaction = st.number_input("Energy-GDP Interaction (combined metric)", value=15000)

# Predict button
if st.button("Predict Carbon Emissions"):
    # Make the prediction with hardcoded inputs for testing
    prediction = predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction)

    if prediction is not None:
        st.subheader(f"Predicted Per Capita Carbon Emissions: {prediction:.2f} tons/year")
    else:
        st.error("Prediction failed. Please ensure the model is loaded correctly.")

# Optional: File uploader for the model (in case it's missing)
st.write("If the model file is missing, you can upload your own 'xgb_model.pkl'.")
uploaded_model = st.file_uploader("Choose a .pkl file for the model", type="pkl")
if uploaded_model is not None:
    with open("xgb_model.pkl", "wb") as f:
        f.write(uploaded_model.read())
    st.success("Model file uploaded successfully. Please refresh the page.")
