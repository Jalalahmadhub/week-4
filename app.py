import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Smart Loan Approval Predictor ", layout="centered")
# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#set Title
st.title(" Smart Loan Approval Predictor ")

# Load encoders
encoder_files = {
    'Gender': 'label_encoder_Gender.pkl',
    'Married': 'label_encoder_Married.pkl',
    'Dependents': 'label_encoder_Dependents.pkl',
    'Education': 'label_encoder_Education.pkl',
    'Self_Employed': 'label_encoder_Self_Employed.pkl',
    'Property_Area': 'label_encoder_Property_Area.pkl',
    'Loan_Status': 'label_encoder_Loan_Status.pkl'  # Optional for decoding result
}
encoders = {}
for key, file in encoder_files.items():
    with open(file, 'rb') as f:
        encoders[key] = joblib.load(f)

# Load scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('Random Forest_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Load original training column order
with open('columns.pkl', 'rb') as f:
    column_order = joblib.load(f)

# Streamlit input form
with st.form("loan_form"):
    st.subheader("Enter Applicant Details")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Term (months)", [360, 180, 240, 300, 120, 84, 60, 36, 12])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Encode categoricals
        Gender = encoders['Gender'].transform([Gender])[0]
        Married = encoders['Married'].transform([Married])[0]
        Dependents = encoders['Dependents'].transform([Dependents])[0]
        Education = encoders['Education'].transform([Education])[0]
        Self_Employed = encoders['Self_Employed'].transform([Self_Employed])[0]
        Property_Area = encoders['Property_Area'].transform([Property_Area])[0]

        # Feature engineering
        total_income = ApplicantIncome + CoapplicantIncome
        log_total_income = np.log(total_income + 1)
        log_loan_amount = np.log(LoanAmount + 1)
        loan_to_income_ratio = LoanAmount / (total_income + 1)

        # Combine all inputs
        input_data = pd.DataFrame([[
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
            Credit_History, Property_Area, total_income, log_total_income,
            log_loan_amount, loan_to_income_ratio
        ]], columns=[
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area', 'total_income', 'log_total_income',
            'log_loan_amount', 'loan_to_income_ratio'
        ])

        # Reorder to match training columns
        input_data = input_data.reindex(columns=column_order)

        # Scale numeric features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = encoders['Loan_Status'].inverse_transform([prediction])[0]

        # Output
        if result == 'Y':
            st.markdown('<div class="prediction-box approved"> Based on the information provided, your loan is likely to be Approved..</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box rejected"> Based on the information provided, your loan is likely to be  Rejected.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
