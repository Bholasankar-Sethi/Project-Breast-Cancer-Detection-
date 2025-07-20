import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and the scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load dataset to get feature names and their example values (min, max)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# --- Streamlit App ---

# Title of the app
st.title("Breast Cancer Detection App 🩺")
st.write(
    "Enter the tumor measurements below to predict whether it is *Malignant* or *Benign*."
)

# Sidebar for user inputs
st.sidebar.header("Tumor Features")

# Create sliders in the sidebar for user input
input_features = {}
for feature in data.feature_names:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    
    input_features[feature] = st.sidebar.slider(
        label=feature, 
        min_value=min_val, 
        max_value=max_val, 
        value=mean_val
    )

# Convert the dictionary of inputs into a NumPy array
input_df = pd.DataFrame([input_features])
input_array = input_df.values

# Prediction button
if st.button("Predict"):
    # Scale the user input
    input_scaled = scaler.transform(input_array)
    
    # Make a prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # Display the result
    st.subheader("Prediction Result")
    prediction_label = "Benign (Harmless)" if prediction[0] == 1 else "Malignant (Harmful)"
    prediction_color = "green" if prediction[0] == 1 else "red"

    st.markdown(f"The model predicts the tumor is: *<span style='color:{prediction_color};'>{prediction_label}</span>*", unsafe_allow_html=True)
    
    st.subheader("Prediction Probability")
    st.write(f"*Benign:* {prediction_proba[0][1]:.2f}")
    st.write(f"*Malignant:* {prediction_proba[0][0]:.2f}")

st.write("---")
st.write("This app uses a machine learning model to predict breast cancer. It is not a substitute for professional medical advice.")