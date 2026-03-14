import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("csat_prediction_model.pkl", "rb"))

# Sidebar Information
st.sidebar.title("Project Information")
st.sidebar.write("""
Model Used: XGBoost  
Total Features: 86  
Dataset: E-commerce Customer Support Data  
Output: Customer Satisfaction Score (CSAT)
""")

# App Title
st.title("Customer Satisfaction Prediction App")

st.write("""
This application predicts **Customer Satisfaction Score (CSAT)** using a trained Machine Learning model.
Enter the feature values below to get the prediction.
""")

# Create list to store features
features = []

st.subheader("Enter Feature Values")

# Generate 86 input fields
for i in range(86):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

# Predict button
if st.button("Predict CSAT Score"):

    # Convert input to numpy array
    input_data = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    score = int(prediction[0])

    # CSAT Meaning
    if score == 1:
        result = "Very Dissatisfied"
    elif score == 2:
        result = "Dissatisfied"
    elif score == 3:
        result = "Neutral"
    elif score == 4:
        result = "Satisfied"
    else:
        result = "Very Satisfied"

    # Display Result
    st.success(f"Predicted CSAT Score: {score} ({result})")