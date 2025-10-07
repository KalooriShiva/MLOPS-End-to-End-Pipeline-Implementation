import streamlit as st
import pandas as pd
import sys
import os

# Add the project's 'src' directory to the Python path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importing the necessary components from your MLOps pipeline
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline
from src.exception import MyException

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Vehicle Insurance Claim Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar for Training ---
with st.sidebar:
    st.header("Model Training")
    st.write("Click the button below to start a new training run for the model.")
    
    if st.button("Train Model"):
        st.info("Training process started. This may take a few minutes...")
        try:
            with st.spinner("Running training pipeline..."):
                train_pipeline = TrainPipeline()
                train_pipeline.run_pipeline()
            st.success("Training successful!")
            st.balloons()
        except Exception as e:
            st.error(f"An error occurred during training: {MyException(e, sys)}")

# --- Main Application Interface ---
st.title("Vehicle Insurance Claim Prediction")
st.write("Enter the customer and vehicle details below to predict whether a claim will be approved.")

# --- User Input Form ---
# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    # Map Gender string to integer (Male: 1, Female: 0)
    gender_str = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_str == "Male" else 0
    
    age = st.number_input("Driver's Age", min_value=18, max_value=100, value=30, step=1)
    
    # The selectbox for Driving License already returns 0 or 1.
    # This format_func makes the display clearer for the user.
    driving_license = st.selectbox("Has Driving License?", [1, 0], format_func=lambda x: "1 - Yes" if x == 1 else "0 - No")
    
    region_code = st.number_input("Region Code", min_value=0, max_value=52, value=28, step=1)

with col2:
    previously_insured = st.selectbox("Previously Insured?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
    vehicle_damage = st.selectbox("Vehicle Previously Damaged?", ["Yes", "No"])
    
with col3:
    annual_premium = st.number_input("Annual Premium (€)", min_value=2000.0, max_value=100000.0, value=30000.0, step=1000.0)
    policy_sales_channel = st.number_input("Policy Sales Channel Code", min_value=1, max_value=200, value=152, step=1)
    vintage = st.number_input("Days as Customer (Vintage)", min_value=10, max_value=300, value=150, step=1)

# --- Prediction Logic ---
if st.button("Predict Claim Status", type="primary"):
    try:
        # --- Perform one-hot encoding based on user input ---
        
        # For Vehicle_Age
        Vehicle_Age_lt_1_Year = 1 if vehicle_age == '< 1 Year' else 0
        Vehicle_Age_gt_2_Years = 1 if vehicle_age == '> 2 Years' else 0
        
        # For Vehicle_Damage
        Vehicle_Damage_Yes = 1 if vehicle_damage == 'Yes' else 0

        # Instantiate the data class with the correctly encoded user inputs
        vehicle_data = VehicleData(
            Gender=gender, # Pass the numerical gender (1 or 0)
            Age=age,
            Driving_License=driving_license,
            Region_Code=region_code,
            Previously_Insured=previously_insured,
            Annual_Premium=annual_premium,
            Policy_Sales_Channel=policy_sales_channel,
            Vintage=vintage,
            # Use the new one-hot encoded variables
            Vehicle_Age_lt_1_Year=Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=Vehicle_Damage_Yes
        )

        # Get the data as a pandas DataFrame
        input_df = vehicle_data.get_vehicle_input_data_frame()
        
        st.write("---")
        st.write("User Input:")
        st.dataframe(input_df)

        # Initialize the prediction pipeline and make a prediction
        with st.spinner("Analyzing..."):
            model_predictor = VehicleDataClassifier()
            prediction = model_predictor.predict(dataframe=input_df)
        
        st.write("---")
        st.subheader("Prediction Result")
        
        # Display the result
        if prediction[0] == 1:
            st.success("Claim is likely to be **APPROVED**")
        else:
            st.error("Claim is likely to be **REJECTED**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {MyException(e, sys)}")