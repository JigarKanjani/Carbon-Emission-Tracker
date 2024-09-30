import streamlit as st
import numpy as np
import joblib
import os

# Load the XGBoost model and StandardScaler
@st.cache_resource
def load_model_and_scaler():
    model_path = "xgb_model.pkl"
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        st.error("Model or Scaler file not found. Please upload 'xgb_model.pkl' and 'scaler.pkl'.")
        return None, None

model, scaler = load_model_and_scaler()

# Function to interpret the impact levels based on prediction
def interpret_impact(prediction):
    if prediction < 5:
        return "Safe: Emission levels are within sustainable limits."
    elif 5 <= prediction < 10:
        return "Moderate: Emissions may cause noticeable environmental impact."
    else:
        return "Unsafe: Emissions are too high, leading to severe consequences."

# Prediction function
def predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction):
    input_data = np.array([[energy_consumption, gdp_ppp, energy_gdp_interaction]])
    
    # Debugging: Inspect input before scaling
    st.write("Input data before scaling:", input_data)
    
    if model is not None and scaler is not None:
        # Apply scaling to the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Debugging: Inspect scaled input data
        st.write("Scaled input data:", input_data_scaled)
        
        # Make prediction using the scaled input data
        prediction = model.predict(input_data_scaled)[0]
        
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

# Input fields for user
energy_consumption = st.number_input("Per Capita Energy (kWh/year)", value=25100)
gdp_ppp = st.number_input("Per Capita GDP (PPP USD/year)", value=52000)
energy_gdp_interaction = st.number_input("Energy-GDP Interaction (combined metric)", value=15000)

# Predict button
if st.button("Predict Carbon Emissions"):
    prediction = predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction)
    
    if prediction is not None:
        impact_text = interpret_impact(prediction)
        st.subheader(f"Predicted Per Capita Carbon Emissions: {prediction:.2f} tons/year")
        st.write(impact_text)
    else:
        st.error("Prediction failed. Please ensure the model and scaler are loaded correctly.")

# File uploader for model and scaler (Optional)
st.write("If the model or scaler is missing, you can upload your own 'xgb_model.pkl' and 'scaler.pkl' files.")
uploaded_model = st.file_uploader("Choose a .pkl file for the model", type="pkl")
if uploaded_model is not None:
    with open("xgb_model.pkl", "wb") as f:
        f.write(uploaded_model.read())
    st.success("Model file uploaded successfully. Please refresh the page.")

uploaded_scaler = st.file_uploader("Choose a .pkl file for the scaler", type="pkl")
if uploaded_scaler is not None:
    with open("scaler.pkl", "wb") as f:
        f.write(uploaded_scaler.read())
    st.success("Scaler file uploaded successfully. Please refresh the page.")
