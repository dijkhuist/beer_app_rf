
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained Random Forest model
with open("rf_beer_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title and description
st.title("🍺 Beer Consumption Predictor - São Paulo")
st.markdown("""
This app predicts daily beer consumption in São Paulo based on weather conditions and weekend status.
Use the sidebar to enter values and simulate consumption.
""")

# Sidebar inputs
st.sidebar.header("Input Parameters")
max_temp = st.sidebar.number_input("Maximum Temperature (°C)", min_value=10.0, max_value=45.0, value=30.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=5.0)
weekend = st.sidebar.selectbox("Is it a weekend?", options=["No", "Yes"])

# Convert weekend to binary
weekend_binary = 1 if weekend == "Yes" else 0

# Prepare input data
input_data = pd.DataFrame({
    'MaxTemp': [max_temp],
    'Rainfall_mm': [rainfall],
    'Weekend': [weekend_binary]
})

# Predict and display result
if st.button("Predict Beer Consumption"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Beer Consumption: {prediction:,.0f} Litres")

# Display historical trend plot
st.markdown("## 📈 Historical Beer Consumption Trend")
st.image("beer_consumption_trend.png")

# Footer
st.markdown("---")
st.caption("Model trained on 2015 data from São Paulo. Developed for educational purposes.")
