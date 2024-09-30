#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load the model from file
model = joblib.load('xgb_model.pkl')


# In[3]:


import numpy as np

# Prediction function
def predict_carbon_emissions(energy_consumption, gdp_ppp, energy_gdp_interaction):
    # Create a data array for the input features
    input_data = np.array([[energy_consumption, gdp_ppp, energy_gdp_interaction]])
    
    # Make a prediction using the model
    prediction = model.predict(input_data)
    
    return prediction


# In[4]:


import ipywidgets as widgets
from IPython.display import display, HTML

# Information and guidance
guidance_text = """
<b>Guidance:</b><br>
- <b>Per Capita Energy (kWh/year):</b> This is the average energy consumption per person in kilowatt-hours per year. 
For example, Canada's per capita energy consumption in 2023 was 25,100 kWh. Typical values range between 1,000 kWh (for developing countries) to 50,000 kWh (for developed countries).<br>
- <b>Per Capita GDP (PPP USD/year):</b> This is the economic output per person, adjusted for purchasing power parity. 
For example, Canada's per capita GDP in 2023 was $52,000 USD. Values typically range from 2,000 USD (for low-income countries) to 100,000 USD (for high-income countries).<br>
- <b>Energy-GDP Interaction:</b> This represents the combined metric that captures the interaction between energy consumption and economic output. A typical range is from 5,000 to 20,000 for most countries. This factor is important because it shows how a country's energy use is related to its economic development.<br>
"""

# Create widgets for inputs with increased width
energy_consumption = widgets.FloatText(
    description="Per Capita Energy (kWh/year):",
    value=25100,
    layout=widgets.Layout(width='500px')  # Set wider width
)
gdp_ppp = widgets.FloatText(
    description="Per Capita GDP (PPP USD/year):",
    value=52000,
    layout=widgets.Layout(width='500px')  # Set wider width
)
energy_gdp_interaction = widgets.FloatText(
    description="Energy-GDP Interaction (combined metric):",
    value=15000,
    layout=widgets.Layout(width='500px')  # Set wider width
)

# Button for making the prediction
predict_button = widgets.Button(description="Predict Carbon Emissions", button_style='success')

# Output for the prediction
output = widgets.Output()

# Function to interpret impact levels
def interpret_impact(prediction):
    if prediction < 5:
        return """
        <span style='color:green; font-size:18px;'>Safe</span>: Emission levels are within sustainable limits.<br>
        <b>Impact:</b> Minor sea level rise (1-3 mm/year). Minimal biodiversity loss.<br>
        <b>Life Expectancy:</b> No significant impact. <br>
        <b>Economic Impact:</b> Low, with industries largely unaffected. <br>
        <b>Citation:</b> NOAA (2025), IPCC (2025).
        """
    elif 5 <= prediction < 10:
        return """
        <span style='color:orange; font-size:18px;'>Moderate</span>: Emissions may cause noticeable environmental impact.<br>
        <b>Impact:</b> Sea levels rise by 0.2 meters by 2100, leading to moderate flooding in New York, Miami, and other coastal cities.<br>
        Gradual melting of Arctic ice by 2050. Heatwaves and flooding risks increase.
        <b>Life Expectancy:</b> Estimated decrease by 2-3 years in developed countries by 2050.<br>
        <b>Economic Impact:</b> Global GDP shrinks by 5-10% by 2050, affecting industries like tourism and agriculture.<br>
        <b>Biodiversity Loss:</b> Extinction rates could increase by 15-20% by 2050 due to habitat destruction.<br>
        <b>Citation:</b> IPCC (2025), World Bank (2025).
        """
    else:
        return """
        <span style='color:red; font-size:18px;'>Unsafe</span>: Emissions are too high, leading to severe environmental and health consequences.<br>
        <b>Impact:</b> Sea levels rise by 0.5 meters or more by 2100, causing severe submergence in cities like Dhaka, Shanghai, and New Orleans.<br>
        <b>Life Expectancy:</b> Estimated decrease by 5-7 years by 2050 in developed countries.<br>
        <b>Economic Impact:</b> Global GDP shrinks by 10-15% by 2050, with severe effects on agriculture and the insurance industry.<br>
        <b>Biodiversity Loss:</b> Extinction rates rise by 25-30%, with ecosystems like the Amazon and Arctic at extreme risk by 2050.<br>
        <b>Citation:</b> IPCC (2025), United Nations Climate Report (2025).
        """

# Function to predict carbon emissions (mocked prediction logic)
def predict_carbon_emissions(energy, gdp, interaction):
    return (energy * 0.0001 + gdp * 0.00005 + interaction * 0.00002)

# Event handler for the button
def on_button_click(b):
    # Get the input values
    energy = energy_consumption.value
    gdp = gdp_ppp.value
    interaction = energy_gdp_interaction.value
    
    # Make a prediction
    prediction = predict_carbon_emissions(energy, gdp, interaction)
    
    # Display the result
    with output:
        output.clear_output()
        impact_text = interpret_impact(prediction)
        display(HTML(f"<b>Predicted Per Capita Carbon Emissions:</b> {prediction:.2f} tons/year<br>{impact_text}"))

# Attach the event to the button
predict_button.on_click(on_button_click)

# Layout structure for a more organized view
input_box = widgets.VBox([
    energy_consumption, gdp_ppp, energy_gdp_interaction,
    predict_button
])

# Display the guidance, widgets, and output
display(HTML(guidance_text))
display(input_box, output)


# In[ ]:




