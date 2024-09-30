import streamlit as st
import numpy as np
import joblib
import os

# Load the model from the pkl file
@st.cache_resource  # Use caching to avoid reloading the model each time
def load_model():
    model_path = "xgb_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found. Please upload 'xgb_model.pkl'.")
        return None

model = load_model()

# Prediction function using the loaded model
def predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction):
    if model is not None:
        # Create a data array for the input features
        input_data = np.array([[energy_consumption, gdp_ppp, energy_gdp_interaction]])
        # Make a prediction using the model
        prediction = model.predict(input_data)[0]  # Get the first value of prediction
        return prediction
    else:
        return None

# Streamlit UI
st.title("Carbon Emissions Prediction")

# Input fields
energy_consumption = st.number_input("Per Capita Energy (kWh/year)", value=25100)
gdp_ppp = st.number_input("Per Capita GDP (PPP USD/year)", value=52000)
energy_gdp_interaction = st.number_input("Energy-GDP Interaction (combined metric)", value=15000)

# Predict button
if st.button("Predict Carbon Emissions"):
    prediction = predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction)
    
    if prediction is not None:
        st.subheader(f"Predicted Per Capita Carbon Emissions: {prediction:.2f} tons/year")
    else:
        st.error("Failed to make a prediction. Ensure the model is loaded correctly.")
