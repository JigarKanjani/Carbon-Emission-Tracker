#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[3]:


import streamlit as st
import numpy as np
import joblib
import os

# Load the XGBoost model
@st.cache_resource  # Use streamlit caching to prevent reloading the model every time
def load_model():
    model_path = "xgb_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found. Please upload 'xgb_model.pkl'.")
        return None

model = load_model()

# Function to interpret impact levels
def interpret_impact(prediction):
    if prediction < 5:
        return "Safe: Emission levels are within sustainable limits."
    elif 5 <= prediction < 10:
        return "Moderate: Emissions may cause noticeable environmental impact."
    else:
        return "Unsafe: Emissions are too high, leading to severe consequences."

# Function for prediction
def predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction):
    input_data = np.array([[energy_consumption, gdp_ppp, energy_gdp_interaction]])
    if model is not None:
        prediction = model.predict(input_data)[0]
        return prediction
    else:
        return None

# Streamlit app layout
st.title("Carbon Emissions Prediction")

st.write("""
Enter your energy consumption, GDP, and energy-GDP interaction to predict per capita carbon emissions.
""")

# Input fields
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
        st.error("Prediction failed. Please ensure the model is loaded correctly.")

# File uploader for model (Optional - in case you want users to upload their own models)
st.write("If the model is missing, you can upload your own 'xgb_model.pkl' file.")
uploaded_file = st.file_uploader("Choose a .pkl file", type="pkl")
if uploaded_file is not None:
    with open("xgb_model.pkl", "wb") as f:
        f.write(uploaded_file.read())
    st.success("Model file uploaded successfully. Please refresh the page.")


# In[ ]:




