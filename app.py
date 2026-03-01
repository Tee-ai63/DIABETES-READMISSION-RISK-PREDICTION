import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json

@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model("model.json")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    return booster, feature_names

booster, ALL_FEATURES = load_model()

st.set_page_config(page_title="Diabetes Readmission Risk", layout="centered")
st.title("🏥 Diabetes Readmission Risk Prediction")
st.subheader("Patient Information")

age = st.selectbox("Age Group", [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)",
    "[40-50)", "[50-60)", "[60-70)", "[70-80)",
    "[80-90)", "[90-100)"
])
gender = st.selectbox("Gender", ["Male", "Female"])
race   = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])

time_in_hospital   = st.slider("Time in Hospital (days)", 1, 14, 3)
num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
num_medications    = st.slider("Number of Medications", 1, 81, 10)
number_diagnoses   = st.slider("Number of Diagnoses", 1, 16, 5)

age_map = {
    '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
    '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}
age_midpoint = age_map[age]

def age_risk_group(a):
    if a in ['[0-10)', '[10-20)', '[20-30)', '[30-40)']:
        return 'Young'
    elif a in ['[40-50)', '[50-60)']:
        return 'Middle_Aged'
    else:
        return 'Senior'

age_group = age_risk_group(age)

if st.button("Predict Readmission Risk"):

    # Start every feature as NaN — XGBoost handles missing values natively
    row = {f: np.nan for f in ALL_FEATURES}

    # Map UI inputs to the exact feature names used during training
    known = {
        "age_midpoint":          float(age_midpoint),
        "time_in_hospital":      float(time_in_hospital),
        "num_lab_procedures":    float(num_lab_procedures),
        "num_medications":       float(num_medications),
        "number_diagnoses":      float(number_diagnoses),
        "age_group_Middle_Aged": 1.0 if age_group == "Middle_Aged" else 0.0,
        "age_group_Senior":      1.0 if age_group == "Senior"      else 0.0,
        "gender_Male":           1.0 if gender == "Male"           else 0.0,
        "race_Asian":            1.0 if race == "Asian"            else 0.0,
        "race_Caucasian":        1.0 if race == "Caucasian"        else 0.0,
        "race_Hispanic":         1.0 if race == "Hispanic"         else 0.0,
        "race_Other":            1.0 if race == "Other"            else 0.0,
    }
    for k, v in known.items():
        if k in row:
            row[k] = v

    input_df  = pd.DataFrame([row], columns=ALL_FEATURES).astype(float)
    dmat      = xgb.DMatrix(input_df)
    risk_prob = float(booster.predict(dmat)[0])

    st.subheader("📊 Prediction Result")
    st.metric("Probability of Readmission", f"{risk_prob * 100:.2f}%")
    if risk_prob >= 0.35:
        st.error("⚠️ High Risk of Readmission")
    else:
        st.success("✅ Low Risk of Readmission")