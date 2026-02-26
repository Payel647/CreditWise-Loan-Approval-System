import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained models and transformers
@st.cache_resource
def load_assets():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

st.set_page_config(page_title="CreditWise - Loan Predictor", layout="wide")

st.title("🏦 CreditWise: Loan Approval Prediction")
st.markdown("Machine Learning (Naive Bayes Classifier)")
st.write("This application predicts loan eligibility based on historical data patterns including credit score, income, and debt-to-income ratio.")
st.markdown("Enter applicant details below to predict loan eligibility.")

# Organize Input Fields
col1, col2, col3 = st.columns(3)

with col1:
    app_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    co_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    dti = st.number_input("DTI Ratio (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.3)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=0)
    savings = st.number_input("Total Savings", min_value=0.0, value=1000.0)

with col3:
    loan_amount = st.number_input("Loan Amount Requested", min_value=0.0, value=15000.0)
    loan_term = st.number_input("Loan Term (In Months)", min_value=12, max_value=360, value=36)
    collateral = st.number_input("Collateral Value", min_value=0.0, value=5000.0)
    edu_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

st.subheader("Additional Demographics")
c1, c2, c3 = st.columns(3)

with c1:
    emp_status = st.selectbox("Employment Status", ['Salaried', 'Self-employed', 'Contract', 'Unemployed'])
    marital = st.selectbox("Marital Status", ['Married', 'Single'])
with c2:
    purpose = st.selectbox("Loan Purpose", ['Personal', 'Car', 'Business', 'Home', 'Education'])
    area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
with c3:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    employer = st.selectbox("Employer Category", ['Private', 'Government', 'MNC', 'Business', 'Unemployed'])

# Prediction Execution
if st.button("Run Prediction", type="primary"):
    # Build dictionary with names matching the original CSV logic
    input_dict = {
        'Applicant_Income': app_income, 
        'Coapplicant_Income': co_income, 
        'Age': age,
        'Dependents': dependents, 
        'Existing_Loans': existing_loans, 
        'Savings': savings,
        'Collateral_Value': collateral, 
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
        'Education_Level': 0 if edu_level == "Graduate" else 1,
        'Employment_Status': emp_status, 
        'Marital_Status': marital, 
        'Loan_Purpose': purpose,
        'Property_Area': area, 
        'Gender': gender, 
        'Employer_Category': employer,
        'DTI_Ratio': dti, 
        'Credit_Score': credit_score
    }
    
    input_df = pd.DataFrame([input_dict])

    # Feature Engineering (Square transformations as per your notebook)
    input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
    input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

    # One-Hot Encoding
    cat_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    encoded_array = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))

    # Combine dataframes and drop raw columns as per notebook logic
    X_final = pd.concat([
        input_df.drop(columns=cat_cols + ['DTI_Ratio', 'Credit_Score']), 
        encoded_df
    ], axis=1)
    
    # RE-ORDERING COLUMNS to match exactly what the Scaler saw during training
    feature_order = [
        'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 
        'Existing_Loans', 'Savings', 'Collateral_Value', 'Loan_Amount', 
        'Loan_Term', 'Education_Level', 'Employment_Status_Salaried', 
        'Employment_Status_Self-employed', 'Employment_Status_Unemployed', 
        'Marital_Status_Single', 'Loan_Purpose_Car', 'Loan_Purpose_Education', 
        'Loan_Purpose_Home', 'Loan_Purpose_Personal', 'Property_Area_Semiurban', 
        'Property_Area_Urban', 'Gender_Male', 'Employer_Category_Government', 
        'Employer_Category_MNC', 'Employer_Category_Private', 
        'Employer_Category_Unemployed', 'DTI_Ratio_sq', 'Credit_Score_sq'
    ]
    
    # This line ensures the order is identical to the training set
    X_final = X_final[feature_order]

    # Scaling and Prediction
    try:
        X_scaled = scaler.transform(X_final)
        prediction = model.predict(X_scaled)

        if prediction[0] == 1:
            st.success(" Prediction Result: **Loan Approved!!**")
        else:
            st.error(" Prediction Result: **Loan Rejected!!**")
            
    except Exception as e:
        st.error(f"Processing Error: {e}")
