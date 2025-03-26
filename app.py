import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import json
import re

# --- Load the Pre-trained ANN Model ---
@st.cache_resource  # Use this if your Streamlit version supports it; otherwise use st.cache(allow_output_mutation=True)
def load_ann_model():
    return tf.keras.models.load_model("ann_model.h5")

try:
    model = load_ann_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- Streamlit UI ---
st.title("App Fraud Detection")
st.write("Enter the app details below:")

# Additional field: App Name (for cross-verification, not used by the ANN)
app_name = st.text_input("App Name", "Enter App Name")
if not app_name.strip() or app_name.strip() == "Enter App Name":
    st.error("Please enter a valid App Name.")
    st.stop()  # Stop execution until a valid app name is provided

# Five feature inputs for the ANN model (numeric features)
rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
rating_count = st.number_input("Rating Count", min_value=0, value=1000, step=1)
installs = st.number_input("Installs", min_value=0, value=5000, step=1)
max_installs = st.number_input("Maximum Installs", min_value=0, value=6000, step=1)
editor_choice = st.selectbox("Editor Choice (1 = Yes, 0 = No)", options=[0, 1], index=1)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded, cannot perform prediction.")
    else:
        # Prepare the ANN input features (exclude App Name)
        input_features = np.array([rating, rating_count, installs, max_installs, editor_choice]).reshape(1, -1)
        
        try:
            prediction_prob = model.predict(input_features)[0, 0]
            
            # Determine preliminary classification:
            if abs(prediction_prob - 0.5) < 0.1:
                prelim_type = "suspected"
            else:
                prelim_type = "fraud" if prediction_prob >= 0.5 else "genuine"
            
            # Continue with the Gemini API call and subsequent logic...
            st.write(f"Preliminary classification: **{prelim_type}** (Probability: {prediction_prob:.4f})")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")

