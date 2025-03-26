# --- Streamlit UI ---
st.title("App Fraud Detection")

st.write("Enter the app details below:")

# Additional field: App Name (for cross-verification, not used by the ANN)
app_name = st.text_input("App Name", "Enter App Name")

# Check if an app name is provided
if not app_name.strip() or app_name.strip() == "Enter App Name":
    st.error("Please enter a valid App Name.")
    st.stop()  # Stop further execution until a valid app name is entered

# Five feature inputs for the ANN model (numeric features)
rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
rating_count = st.number_input("Rating Count", min_value=0, value=1000, step=1)
installs = st.number_input("Installs", min_value=0, value=5000, step=1)
max_installs = st.number_input("Maximum Installs", min_value=0, value=6000, step=1)
editor_choice = st.selectbox("Editor Choice (1 = Yes, 0 = No)", options=[0, 1], index=1)

if st.button("Predict") and model is not None:
    # Prepare the ANN input features (exclude App Name)
    input_features = np.array([rating, rating_count, installs, max_installs, editor_choice]).reshape(1, -1)
    
    # Make prediction
    try:
        prediction_prob = model.predict(input_features)[0, 0]
        
        # Determine preliminary classification:
        if abs(prediction_prob - 0.5) < 0.1:
            prelim_type = "suspected"
        else:
            prelim_type = "fraud" if prediction_prob >= 0.5 else "genuine"
        
        # Build prompt for Gemini API including the App Name for cross-verification
        prompt = (
            f"Given the following app details:\n"
            f"- App Name: {app_name}\n"
            f"- Rating: {rating}\n"
            f"- Rating Count: {rating_count}\n"
            f"- Installs: {installs}\n"
            f"- Maximum Installs: {max_installs}\n"
            f"- Editor Choice: {editor_choice}\n\n"
            f"The preliminary classification is '{prelim_type}'. "
            f"First, analyze the App Name '{app_name}' for any suspicious patterns or known fraud indicators. "
            f"Then provide a concise explanation (up to 300 characters) for why this app is classified as "
            f"'{prelim_type}', taking into account both the app name analysis and the other metrics. "
            "Return the answer strictly in the following JSON format with no markdown formatting or code blocks:\n"
            '{ "type": "fraud"|"genuine"|"suspected", "app_name_analysis": "Brief analysis of app name (is it suspicious?)", '
            '"reason": "Concise explanation including app name considerations (300 char max)" }'
        )
        
        # Use hardcoded Gemini API key
        gemini_api_key = "AIzaSyBZzhpMnPx-YZIWET4v_2Qzt2A5EpmDBIw"  # Your Gemini API key
        
        gemini_response = call_gemini(prompt, gemini_api_key)
        
        # Parse the Gemini API response
        if gemini_response:
            try:
                candidate = gemini_response.get("candidates", [{}])[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [{}])
                gemini_text = parts[0].get("text", "")
                
                with st.expander("Show raw Gemini response"):
                    st.code(gemini_text)
                
                json_text = extract_json_from_text(gemini_text)
                gemini_output = json.loads(json_text)
                
                st.write("Final Output from Gemini:")
                st.json(gemini_output)
            except Exception as e:
                st.error(f"Failed to parse Gemini output: {e}")
                st.text("Raw response:")
                st.code(gemini_text if 'gemini_text' in locals() else "No text received")
        else:
            st.error("No valid response received from Gemini API.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
