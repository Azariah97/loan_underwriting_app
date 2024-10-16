import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Parsing functions for PDF text content
def parse_payslip(text):
    return {
        "Name": "John Doe" if "John Doe" in text else "Not Found",
        "NRC": "123456/78/9" if "123456/78/9" in text else "Not Found",
        "Income": 5000 if "Monthly Salary" in text else "Not Found"
    }

def parse_bank_statement(text):
    return {
        "Deposits": [1000, 2000, 3000, 5000] if "Deposit" in text else [],
        "Loans": [1500] if "Loan" in text else []
    }

def parse_nrc(text):
    return {
        "NRC": "123456/78/9" if "123456/78/9" in text else "Not Found",
        "Name": "John Doe" if "John Doe" in text else "Not Found"
    }

# Verification function
def verify_documents(payslip_data, bank_data, nrc_data):
    results = []
    if payslip_data["Name"] != nrc_data["Name"]:
        results.append("Name mismatch between payslip and NRC.")
    if payslip_data["Income"] not in bank_data["Deposits"]:
        results.append("Income from payslip not found in bank deposits.")
    results.append("Loan detected in bank statement." if bank_data["Loans"] else "No loans detected.")
    return results or ["All documents verified successfully."]

# Machine Learning Model Setup
def train_model():
    np.random.seed(42)
    num_samples = 100
    data = pd.DataFrame({
        'Age': np.random.randint(18, 75, num_samples),
        'Income': np.random.randint(5000, 100000, num_samples),
        'Loan Amount': np.random.randint(1000, 50000, num_samples),
        'Loan Term': np.random.choice([5, 10, 15, 20, 25, 30], num_samples),
        'Credit Score': np.random.randint(300, 850, num_samples),
        'Employment Years': np.random.randint(0, 40, num_samples),
        'Purpose_Home': np.random.choice([1, 0], num_samples),
        'Purpose_Car': np.random.choice([1, 0], num_samples),
        'Purpose_Education': np.random.choice([1, 0], num_samples),
        'Purpose_Business': np.random.choice([1, 0], num_samples),
        'Purpose_Other': np.random.choice([1, 0], num_samples),
        'Approval': np.random.choice([1, 0], num_samples)
    })

    X = data.drop("Approval", axis=1)
    y = data["Approval"]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    pipeline.fit(X, y)
    return pipeline

model = train_model()

# Streamlit UI
st.title("AI Document Verification and Loan Underwriting Assistant")

# File upload
payslip_file = st.file_uploader("Upload Payslip (PDF)", type="pdf")
bank_statement_file = st.file_uploader("Upload Bank Statement (PDF)", type="pdf")
nrc_file = st.file_uploader("Upload NRC (PDF)", type="pdf")

# Loan application inputs
age = st.slider("Age", 18, 75, 30)
income = st.number_input("Monthly Income (ZMW)", value=5000, step=1000)
loan_amount = st.number_input("Loan Amount Requested (ZMW)", value=1000, step=500)
loan_term = st.selectbox("Loan Term (years)", [1, 2, 3, 4, 5])
credit_score = st.slider("Credit Score", 300, 850, 650)
employment_years = st.slider("Years of Employment", 0, 40, 5)
purpose = st.selectbox("Purpose of Loan", ["Home", "Car", "Education", "Business", "Other"])

# Button to process verification and approval
if st.button("Verify and Check Loan Status"):
    if payslip_file and bank_statement_file and nrc_file:
        payslip_text = extract_text_from_pdf(payslip_file)
        bank_statement_text = extract_text_from_pdf(bank_statement_file)
        nrc_text = extract_text_from_pdf(nrc_file)

        payslip_data = parse_payslip(payslip_text)
        bank_data = parse_bank_statement(bank_statement_text)
        nrc_data = parse_nrc(nrc_text)

        # Document verification results
        verification_results = verify_documents(payslip_data, bank_data, nrc_data)
        st.subheader("Document Verification Results")
        for result in verification_results:
            st.write("- " + result)
        
        # Rejection conditions
        if "Name mismatch between payslip and NRC." in verification_results or "Income from payslip not found in bank deposits." in verification_results:
            st.subheader("Loan Application Status")
            st.write("**Loan Status:** Rejected due to document inconsistencies.")
        elif loan_amount > 5 * income:
            st.subheader("Loan Application Status")
            st.write("**Loan Status:** Rejected - Requested loan amount is too high relative to income.")
        else:
            # Prepare input data for ML model with all expected columns
            purpose_one_hot = {
                "Purpose_Home": int(purpose == "Home"),
                "Purpose_Car": int(purpose == "Car"),
                "Purpose_Education": int(purpose == "Education"),
                "Purpose_Business": int(purpose == "Business"),
                "Purpose_Other": int(purpose == "Other"),
            }
            
            # Define all columns expected by the model
            all_columns = [
                "Age", "Income", "Loan Amount", "Loan Term", "Credit Score", 
                "Employment Years", "Purpose_Home", "Purpose_Car", 
                "Purpose_Education", "Purpose_Business", "Purpose_Other"
            ]
            
            # Create input data and ensure all columns are included
            input_data = pd.DataFrame({
                "Age": [age],
                "Income": [income],
                "Loan Amount": [loan_amount],
                "Loan Term": [loan_term],
                "Credit Score": [credit_score],
                "Employment Years": [employment_years],
                **purpose_one_hot
            })

            # Reindex to ensure input_data has all the expected columns
            input_data = input_data.reindex(columns=all_columns, fill_value=0)

            # Predict loan approval
            prediction = model.predict(input_data)[0]
            loan_status = "Approved" if prediction == 1 else "Rejected"
            st.subheader("Loan Application Status")
            st.write(f"**Loan Status:** {loan_status}")
    else:
        st.warning("Please upload all required documents.")
