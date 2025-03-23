# Google Play Fraud Detection & Explanation

This repository provides an end-to-end solution for detecting fraudulent apps on the Google Play Store. The project combines data preprocessing, rule-based labeling, and a supervised Artificial Neural Network (ANN) for classification. Additionally, it integrates with the Gemini API to generate natural language explanations, delivering the final output in a strict JSON format.

> **Final Output Format:**  
> ```json
> { "type": "fraud"|"genuine"|"suspected", "reason": "Concise explanation (300 char max)" }
> ```

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Supervised ANN Model Training](#supervised-ann-model-training)
- [Streamlit Web Application & Gemini API Integration](#streamlit-web-application--gemini-api-integration)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project aims to detect fraudulent Google Play apps by combining:

1. **Data Preprocessing & EDA:**  
   - Clean and transform raw Google Play data.
   - Handle missing values and convert string-based numeric fields (e.g., "Installs", "Maximum Installs") to numeric.
   - Encode categorical fields (such as "Editors Choice" and "Developer ID") while retaining reference columns like "App Name" and "Category".

2. **Feature Engineering & Rule-Based Labeling:**  
   - Generate a supervised target (`final_label`), where **0** represents genuine apps and **1** represents fraud, based on domain-specific rules.

3. **Supervised ANN Model Training:**  
   - Train an ANN using TensorFlow/Keras on key features: `Rating`, `Rating Count`, `Installs`, `Maximum Installs`, and `Editor Choice`.
   - Evaluate the model via cross-validation and save the final model as `ann_model.h5`.

4. **Streamlit Web App & Gemini API:**  
   - The interactive app accepts user input for the five feature columns plus "App Name" (used for cross-verification).
   - The ANN model produces a preliminary classification (fraud, genuine, or suspected) based on a threshold.
   - All input details (including App Name) are sent to the Gemini API, which returns a concise explanation in a JSON format as specified.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/arko-14/google-play-data.git
   cd google-play-data
2. **Create a Virtual Environment (Optional but Recommended):**

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
Install Dependencies:
pip install -r requirements.txt

## requirements.txt:
streamlit
tensorflow
numpy
pandas
requests

## Data Preprocessing and Feature Engineering
Data Cleaning & EDA:
Refer to notebooks/EDA.ipynb for exploratory analysis, cleaning of columns (e.g., converting "Installs" and "Maximum Installs" to numeric), and visualization.

## Feature Engineering:
Use notebooks/feature_engineering.ipynb to:

Encode columns such as "Editors Choice" (0/1) and "Developer ID".

Generate a rule-based target (final_label), where 0 = genuine and 1 = fraud.

Retain "App Name" and "Category" for reference (but exclude them from the training feature set).

Supervised ANN Model Training
The notebook notebooks/supervised_ann.ipynb trains the ANN model using the features:

Rating, Rating Count, Installs, Maximum Installs, and Editor Choice

(Optionally: DeveloperID_encoded and Category_encoded)

The model is built with TensorFlow/Keras and evaluated using cross-validation.

The final trained model is saved as ann_model.h5.

## Streamlit Web Application
The [app.py](http://app.py/) file is an interactive web application built with Streamlit. It performs the following:

Model Loading:
Loads the pre-trained ANN model from ann_model.h5.

User Inputs:
Provides input fields for:

App Name: For cross-verification with the Gemini API.

Rating, Rating Count, Installs, Maximum Installs, Editor Choice:
These five numeric features are used by the ANN for preliminary prediction.

ANN Prediction:
The model outputs a preliminary classification:

Uses a threshold of 0.5 (with a near-threshold range designated as "suspected").

Gemini API Integration:
The app constructs a prompt with all details (including App Name) and sends it to the Gemini API.
The Gemini API returns a final output in JSON format:

## { "type": "fraud"|"genuine"|"suspected", "reason": "Concise explanation (300 char max)" }
Display Output:
The final JSON output from Gemini is displayed in the app.


