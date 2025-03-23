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



