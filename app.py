import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import re

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Risk",
    page_icon="🏥",
    layout="centered"
)

# ──────────────────────────────────────────────
# CUSTOM STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { 
        background-color: #1e3a5f !important; 
        padding: 1rem; 
        border-radius: 8px;
        border-left: 4px solid #60a5fa; 
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #93c5fd !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricDelta"] {
        color: #fbbf24 !important;
    }
    .causal-box { 
        background-color: #1e3a5f; 
        border: 1px solid #3b82f6;
        border-radius: 8px; 
        padding: 1rem; 
        margin-top: 1rem;
        color: #ffffff;
    }
    .warning-box { 
        background-color: #78350f; 
        border: 1px solid #fcd34d;
        border-radius: 8px; 
        padding: 1rem; 
        margin-top: 0.5rem;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model("model.json")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    return booster, feature_names

booster, ALL_FEATURES = load_model()

def sanitize(name):
    """Replace characters XGBoost forbids in feature names."""
    return re.sub(r'[\[\]<>\s]', '_', str(name))

CLEAN_FEATURES = [sanitize(f) for f in ALL_FEATURES]
FEATURE_MAP    = {orig: clean for orig, clean in zip(ALL_FEATURES, CLEAN_FEATURES)}


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def age_risk_group(age_range):
    if age_range in ['[0-10)', '[10-20)', '[20-30)', '[30-40)']:
        return 'Young'
    elif age_range in ['[40-50)', '[50-60)']:
        return 'Middle_Aged'
    else:
        return 'Senior'


def compute_uplift_score(risk_prob, number_inpatient, discharge_destination,
                          number_diagnoses, num_medications):
    """
    Causal ML — Uplift Score Estimation.

    A predictive model tells you WHO is likely to be readmitted.
    An uplift score estimates WHO would actually BENEFIT from intervention.

    These are not the same thing:
    - A patient with severe uncontrolled disease may be high-risk but
      will be readmitted regardless of intervention (low uplift).
    - A patient with manageable risk factors may respond strongly to
      a follow-up call or medication review (high uplift).

    Methodology:
    We approximate uplift using a clinically-motivated modifiability index.
    Features are split into:
      - Fixed/unmodifiable: number_inpatient, number_diagnoses
        (proxy for underlying disease severity — hard to change quickly)
      - Modifiable: discharge destination, medication complexity
        (factors that quality discharge planning can directly address)

    Uplift score = risk_prob × modifiability_weight
    High uplift = patient is at risk AND their risk drivers are actionable.
    Low uplift  = patient is at risk but risk is driven by fixed disease burden.
    """

    # Modifiability weight — how much of this patient's risk is actionable?
    # Discharge to a complex destination signals intervention opportunity
    discharge_modifiability = {
        "Home":                      0.3,   # low — already going home, stable
        "Home with Health Service":  0.7,   # high — can improve home care quality
        "Skilled Nursing Facility":  0.6,   # moderate — can coordinate handoff
        "Rehabilitation":            0.5,   # moderate — structured setting
        "Other":                     0.4,
    }
    d_weight = discharge_modifiability.get(discharge_destination, 0.4)

    # High prior inpatient visits = deep disease severity = lower modifiability
    # This is the confounded feature — not a cause, a proxy for severity
    severity_penalty = min(number_inpatient / 10.0, 0.5)

    # High medication complexity = intervention opportunity
    # Medication review at discharge is a proven readmission reducer
    med_opportunity = min(num_medications / 20.0, 0.4)

    modifiability = (d_weight + med_opportunity - severity_penalty)
    modifiability = max(0.1, min(modifiability, 1.0))  # clamp to [0.1, 1.0]

    uplift = risk_prob * modifiability
    return round(uplift, 4), round(modifiability, 4)


def get_intervention_recommendation(risk_prob, uplift_score, number_inpatient,
                                     discharge_destination, num_medications,
                                     number_diagnoses):
    """
    Translate risk + uplift into a concrete clinical recommendation.
    This is the bridge between prediction and action.
    """
    recommendations = []

    if risk_prob >= 0.4285 and uplift_score >= 0.25:
        recommendations.append(
            "🔴 **High Priority** — Flag for care coordinator review before discharge. "
            "This patient is high risk and their risk drivers appear actionable."
        )
    elif risk_prob >= 0.4285 and uplift_score < 0.25:
        recommendations.append(
            "🟠 **Monitor** — High risk but risk is primarily driven by underlying "
            "disease severity. Standard discharge protocol with documented follow-up plan."
        )
    else:
        recommendations.append(
            "🟢 **Standard Discharge** — Below readmission risk threshold. "
            "Routine follow-up recommended."
        )

    # Specific intervention signals
    if num_medications >= 10:
        recommendations.append(
            "💊 **Medication Review** — High medication count detected. "
            "Pharmacist reconciliation at discharge is recommended."
        )
    if discharge_destination in ["Home with Health Service", "Skilled Nursing Facility"]:
        recommendations.append(
            "🤝 **Handoff Coordination** — Complex discharge destination. "
            "Ensure receiving facility has full discharge summary and medication list."
        )
    if number_inpatient >= 3:
        recommendations.append(
            "📋 **Disease Management** — Multiple prior inpatient stays suggest "
            "chronic disease instability. Consider specialist referral or "
            "enhanced diabetes management programme."
        )
    if number_diagnoses >= 7:
        recommendations.append(
            "🔬 **Comorbidity Review** — High diagnosis count. "
            "Confirm all conditions are addressed in discharge plan."
        )

    return recommendations


# ──────────────────────────────────────────────
# APP HEADER
# ──────────────────────────────────────────────
st.title("🏥 Diabetes Readmission Risk Predictor")
st.markdown(
    "Predicts 30-day hospital readmission risk for diabetic patients "
    "and estimates which patients would most benefit from discharge intervention."
)
st.caption(
    "Built on the UCI Diabetes 130-US Hospitals dataset (101,766 records). "
    "Threshold calibrated at 0.4285 for ≥ 80% recall. "
    "This is a **screening tool** — not a clinical diagnosis."
)
st.divider()


# ──────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age Group", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)",
        "[40-50)", "[50-60)", "[60-70)", "[70-80)",
        "[80-90)", "[90-100)"
    ], index=6)

    gender = st.selectbox("Gender", ["Male", "Female"])

    race = st.selectbox("Race", [
        "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
    ])

    discharge_destination = st.selectbox("Discharge Destination", [
        "Home",
        "Home with Health Service",
        "Skilled Nursing Facility",
        "Rehabilitation",
        "Other"
    ])

with col2:
    number_inpatient = st.slider(
        "Prior Inpatient Visits (last year)", 0, 20, 0,
        help="Number of inpatient admissions in the year before this encounter. "
             "Top predictive feature — proxy for disease severity."
    )

    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)

    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)

    num_medications = st.slider(
        "Number of Medications", 1, 81, 10,
        help="Total medications administered during this encounter."
    )

    number_diagnoses = st.slider(
        "Number of Diagnoses", 1, 16, 5,
        help="Number of diagnoses recorded. Higher values indicate comorbidity burden."
    )

st.divider()


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
if st.button("🔍 Predict Readmission Risk", use_container_width=True, type="primary"):

    # ── Build feature row ──────────────────────
    age_map = {
        '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
        '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    age_midpoint = age_map[age]
    age_group    = age_risk_group(age)

    # Discharge destination → discharge_disposition_id proxy
    disposition_map = {
        "Home":                      1,
        "Home with Health Service":  6,
        "Skilled Nursing Facility":  3,
        "Rehabilitation":            4,
        "Other":                     5,
    }
    discharge_disposition_id = disposition_map.get(discharge_destination, 1)

    row = {sanitize(f): np.nan for f in ALL_FEATURES}

    known = {
        "age_midpoint":              float(age_midpoint),
        "time_in_hospital":          float(time_in_hospital),
        "num_lab_procedures":        float(num_lab_procedures),
        "num_medications":           float(num_medications),
        "number_diagnoses":          float(number_diagnoses),
        "number_inpatient":          float(number_inpatient),
        "discharge_disposition_id":  float(discharge_disposition_id),
        "age_group_Middle_Aged":     1.0 if age_group == "Middle_Aged" else 0.0,
        "age_group_Senior":          1.0 if age_group == "Senior"      else 0.0,
        "gender_Male":               1.0 if gender == "Male"           else 0.0,
        "race_Asian":                1.0 if race == "Asian"            else 0.0,
        "race_Caucasian":            1.0 if race == "Caucasian"        else 0.0,
        "race_Hispanic":             1.0 if race == "Hispanic"         else 0.0,
        "race_Other":                1.0 if race == "Other"            else 0.0,
    }
    for k, v in known.items():
        if k in row:
            row[k] = v

    input_df  = pd.DataFrame([row], columns=CLEAN_FEATURES).astype(float)
    dmat      = xgb.DMatrix(input_df)
    risk_prob = float(booster.predict(dmat)[0])

    # ── Causal ML: Uplift Score ────────────────
    uplift_score, modifiability = compute_uplift_score(
        risk_prob        = risk_prob,
        number_inpatient = number_inpatient,
        discharge_destination = discharge_destination,
        number_diagnoses = number_diagnoses,
        num_medications  = num_medications
    )

    # ── Display Results ────────────────────────
    st.subheader("📊 Prediction Results")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            label="Readmission Probability",
            value=f"{risk_prob * 100:.1f}%",
            delta="Above threshold" if risk_prob >= 0.4285 else "Below threshold",
            delta_color="inverse"
        )

    with col_b:
        st.metric(
            label="Uplift Score",
            value=f"{uplift_score * 100:.1f}%",
            help="Causal ML estimate: how much would this patient benefit from intervention?"
        )

    with col_c:
        st.metric(
            label="Modifiability Index",
            value=f"{modifiability * 100:.0f}%",
            help="How actionable are this patient's risk drivers? "
                 "High = risk is driven by factors discharge planning can address."
        )

    st.caption(f"Risk threshold: **42.85%** — patients above this are flagged for review.")

    # ── Risk Banner ────────────────────────────
    st.divider()
    if risk_prob >= 0.4285:
        st.error(
            "⚠️ **HIGH RISK** — This patient exceeds the readmission risk threshold. "
            "Recommend care coordinator review before discharge."
        )
    else:
        st.success(
            "✅ **LOW RISK** — This patient is below the readmission risk threshold. "
            "Standard discharge protocol applies."
        )

    # ── Causal ML Section ─────────────────────
    st.divider()
    st.subheader("🧠 Causal ML: Intervention Targeting")

    st.markdown("""
    > **Prediction tells you *who* is at risk. Causal ML tells you *who would benefit* from action.**
    >
    > Not every high-risk patient responds equally to intervention. A patient whose
    > readmission risk is driven entirely by severe chronic disease may be readmitted
    > regardless of what the care team does at discharge. A patient whose risk is
    > driven by medication complexity or discharge destination may respond strongly
    > to targeted support.
    >
    > The **Uplift Score** estimates which type of patient this is.
    """)

    # Uplift interpretation
    if uplift_score >= 0.30:
        st.markdown(
            f"""<div class='causal-box'>
            <strong>🎯 High Intervention Potential (Uplift: {uplift_score*100:.1f}%)</strong><br>
            This patient's risk drivers appear largely modifiable. Targeted discharge
            interventions — medication review, follow-up scheduling, handoff coordination —
            are likely to meaningfully reduce readmission probability.
            </div>""",
            unsafe_allow_html=True
        )
    elif uplift_score >= 0.15:
        st.markdown(
            f"""<div class='warning-box'>
            <strong>⚡ Moderate Intervention Potential (Uplift: {uplift_score*100:.1f}%)</strong><br>
            Some risk drivers are modifiable. Targeted support is recommended but
            outcomes will also depend on underlying disease management.
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class='causal-box'>
            <strong>📌 Low Intervention Potential (Uplift: {uplift_score*100:.1f}%)</strong><br>
            Risk appears primarily driven by underlying disease severity — a fixed factor
            that discharge planning cannot directly change. Standard protocol with documented
            long-term disease management referral recommended.
            </div>""",
            unsafe_allow_html=True
        )

    # ── Intervention Recommendations ──────────
    st.divider()
    st.subheader("📌 Clinical Recommendations")

    recommendations = get_intervention_recommendation(
        risk_prob            = risk_prob,
        uplift_score         = uplift_score,
        number_inpatient     = number_inpatient,
        discharge_destination= discharge_destination,
        num_medications      = num_medications,
        number_diagnoses     = number_diagnoses
    )

    for rec in recommendations:
        st.markdown(f"- {rec}")

    # ── Causal Note ────────────────────────────
    st.divider()
    st.caption(
        "**Causal ML note:** `number_inpatient` is the model's top predictive feature "
        "but is treated as a proxy for disease severity, not a cause of readmission. "
        "Both frequent hospitalisation and readmission are caused by the same underlying "
        "disease burden. The uplift score accounts for this by down-weighting prior "
        "inpatient visits in the modifiability calculation. "
        "Reducing admissions would not reduce readmission risk — addressing disease "
        "severity would."
    )

    st.caption(
        "⚠️ This tool is a **clinical screening aid**, not a diagnostic instrument. "
        "All flagged patients should be reviewed by a qualified care coordinator. "
        "Model trained on UCI Diabetes 130-US Hospitals dataset (1999–2008). "
        "External validation on your patient population is recommended before deployment."
    )