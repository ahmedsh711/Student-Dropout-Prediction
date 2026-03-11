import streamlit as st
import pandas as pd
import numpy as np
import joblib

model     = joblib.load('model.pkl')
scaler    = joblib.load('scaler.pkl')
features  = joblib.load('features.pkl')
threshold = joblib.load('threshold.pkl')

label_map = {0: "🔴 Dropout", 1: "🟡 Enrolled", 2: "🟢 Graduate"}

st.title("Student Dropout Predictor")
st.write("Fill in the student's information to predict their outcome.")

st.subheader("Academic Performance")
approval_rate     = st.slider("Approval Rate (approved / enrolled)", 0.0, 1.0, 0.7)
avg_approved      = st.number_input("Average Approved Units", 0, 30, 10)
sem2_approved_raw = st.number_input("Sem 2 Approved Units", 0, 30, 8)
avg_grade         = st.number_input("Average Grade", 0.0, 20.0, 12.0)
grade_trend       = st.number_input("Grade Trend (Sem2 - Sem1)", -10.0, 10.0, 0.0)

st.subheader("Financial Status")
tuition_up_to_date = st.selectbox("Tuition Fees Up to Date?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
scholarship        = st.selectbox("Has Scholarship?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
debtor             = st.selectbox("Is a Debtor?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.subheader("Enrollment Info")
age               = st.number_input("Age at Enrollment", 17, 60, 20)
application_order = st.number_input("Application Order (1 = first choice)", 1, 10, 1)

financial_risk   = int(scholarship == 0 and tuition_up_to_date == 0)
not_first_choice = int(application_order > 1)

input_data = {
    'approval_rate':            approval_rate,
    'avg_approved':             avg_approved,
    'sem2_approved_raw':        sem2_approved_raw,
    'avg_grade':                avg_grade,
    'grade_trend':              grade_trend,
    'Tuition fees up to date':  tuition_up_to_date,
    'Scholarship holder':       scholarship,
    'Debtor':                   debtor,
    'Age at enrollment':        age,
    'financial_risk':           financial_risk,
    'not_first_choice':         not_first_choice,
}

input_df     = pd.DataFrame([input_data])
input_df     = input_df.reindex(columns=features, fill_value=0)
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    proba = model.predict_proba(input_scaled)[0]

    # apply calibrated threshold for Enrolled (class 1)
    pred_key = int(np.argmax(proba))
    if proba[1] >= threshold:
        pred_key = 1

    st.markdown("---")
    st.subheader(f"Prediction: {label_map[pred_key]}")

    st.write("**Confidence breakdown:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dropout",  f"{proba[0]*100:.1f}%")
    col2.metric("Enrolled", f"{proba[1]*100:.1f}%")
    col3.metric("Graduate", f"{proba[2]*100:.1f}%")
