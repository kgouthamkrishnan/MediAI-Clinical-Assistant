import streamlit as st
import numpy as np
import joblib
import datetime 

# PAGE CONFIG
st.set_page_config(
    page_title="MediAI Treatment Predictor",
    page_icon="🩺",
    layout="wide"
)

# LOAD MODELS
try:
    model = joblib.load("models/healthcare_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_sex = joblib.load("models/le_sex.pkl")
    le_bp = joblib.load("models/le_bp.pkl")
    le_chol = joblib.load("models/le_chol.pkl")
    le_sugar = joblib.load("models/le_sugar.pkl")
    le_drug = joblib.load("models/le_drug.pkl")
except Exception as e:
    st.error(f"Model Loading Error: {e}")

# COMPACT CYBER CSS & SIDEBAR FIX
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Main Container Padding */
.block-container {
    padding-top: 5.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 95% !important;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
    radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 25%),
    radial-gradient(circle at bottom right, rgba(6,182,212,0.12), transparent 25%),
    linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white !important;
}

/* SIDEBAR ULTRA-COMPACTNESS FIX */
section[data-testid="stSidebar"] div.stVerticalBlock {
    gap: 0.35rem !important;
}
section[data-testid="stSidebar"] .stSlider {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    margin-bottom: -4px !important;
}
section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stTextInput {
    margin-bottom: -4px !important;
}
section[data-testid="stSidebar"] label p {
    margin-bottom: 1px !important;
    font-size: 13px !important;
}

/* HEADER */
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: white;
    margin-top: 5px;
    margin-bottom: 2px;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 25px;
}

/* GLASS CARD */
.glass {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    height: 100%;
}

/* CLINICAL LIST ITEMS */
.profile-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.04);
}
.profile-label { color: #cbd5e1; font-weight: 500; font-size: 14px; }
.profile-value { color: #3b82f6; font-weight: 600; font-size: 14px; background: rgba(59,130,246,0.1); padding: 2px 10px; border-radius: 6px; }

/* COMPACT BUTTON */
.stButton>button, .stDownloadButton>button {
    width: 100%;
    height: 42px;
    border: none !important;
    border-radius: 10px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
    color: white !important;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(59,130,246,0.25);
    margin-top: 10px !important;
    transition: 0.2s;
}
.stButton>button:hover, .stDownloadButton>button:hover { transform: scale(1.01); }

/* COMPACT RESULT BOX */
.result-box {
    background: linear-gradient(135deg, #064e3b, #14532d);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(34,197,94,0.3);
    margin-top: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.result-title { font-size: 15px; color: #a7f3d0; font-weight: 600; }
.result-drug { font-size: 30px; font-weight: 800; color: white; margin-top: 4px; margin-bottom: 10px; }
.result-advice { background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; font-size: 13px; color: #e2e8f0; line-height: 1.5; text-align: left; border-left: 4px solid #34d399; }

/* INPUT FIXES */
div[data-baseweb="select"] > div { background-color: rgba(255, 255, 255, 0.05) !important; color: white !important; }
div[data-baseweb="popover"] ul { background-color: #0f172a !important; }
div[data-baseweb="popover"] li { color: white !important; }
section[data-testid="stSidebar"] { background-color: #020617 !important; }
input { background-color: rgba(255, 255, 255, 0.05) !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-title">🩺 MediAI Clinical Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Powered Smart Healthcare Recommendation System</div>', unsafe_allow_html=True)

# SIDEBAR CLINICAL INPUTS
with st.sidebar:
    st.markdown("<h3 style='margin-top:0; margin-bottom:5px; font-size:18px; color:white;'>⚙️ Clinical Inputs</h3>", unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", "John Doe")
    age = st.slider("Age", 15, 100, 45)
    sex = st.selectbox("Gender", ["F", "M"])
    bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
    sugar = st.selectbox("Sugar Level", ["NORMAL", "PRE_DIABETIC", "DIABETIC"])
    bmi = st.slider("BMI", 15.0, 45.0, 23.5, step=0.1)
    na_to_k = st.slider("Na_to_K Ratio", 5.0, 40.0, 12.0, step=0.1)
    
    predict_btn = st.button("Generate Recommendation")

# MAIN WORKSPACE
left, right = st.columns([1, 1.1])

# LEFT PANEL: Patient Clinical Profile
with left:
    st.markdown(f"""
    <div class="glass">
        <h3 style='margin:0 0 12px 0; color:white; font-size:18px;'>📋 Patient Clinical Profile</h3>
        <div class="profile-item"><span class="profile-label">👤 Patient Name</span><span class="profile-value" style="color:#10b981;">{patient_name}</span></div>
        <div class="profile-item"><span class="profile-label">🗓️ Age</span><span class="profile-value">{age}</span></div>
        <div class="profile-item"><span class="profile-label">⚥ Gender</span><span class="profile-value">{sex}</span></div>
        <div class="profile-item"><span class="profile-label">🫀 Blood Pressure</span><span class="profile-value">{bp}</span></div>
        <div class="profile-item"><span class="profile-label">❤️ Cholesterol</span><span class="profile-value">{chol}</span></div>
        <div class="profile-item"><span class="profile-label">⚖️ BMI</span><span class="profile-value">{bmi}</span></div>
        <div class="profile-item"><span class="profile-label">🍬 Sugar Status</span><span class="profile-value">{sugar}</span></div>
        <div class="profile-item"><span class="profile-label">🧪 Na_to_K Ratio</span><span class="profile-value">{na_to_k}</span></div>
    </div>
    """, unsafe_allow_html=True)

# RIGHT PANEL: AI Diagnostics Hub & Results
with right:
    st.markdown("""<div class="glass"><div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;"><div style="width:36px; height:36px; border-radius:8px; background:linear-gradient(135deg, #06b6d4, #3b82f6); display:flex; align-items:center; justify-content:center; font-size:18px;">🤖</div><div><div style="font-size:16px; font-weight:700; color:white;">AI Diagnostics Hub</div><div style="color:#94a3b8; font-size:11px;">Real-time target features monitoring</div></div></div><div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; color:#cbd5e1;"><div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px;">✅ Age Analysis</div><div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px;">🫀 BP Monitoring</div><div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px;">⚖️ BMI Evaluation</div><div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px;">❤️ Cholesterol Check</div><div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px;">🍬 Sugar Detection</div><div style="background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px;">🧪 Electrolyte Balance</div></div>""", unsafe_allow_html=True)

    if predict_btn:
        try:
            sex_encoded = le_sex.transform([sex])[0]
            bp_encoded = le_bp.transform([bp])[0]
            chol_encoded = le_chol.transform([chol])[0]
            sugar_encoded = le_sugar.transform([sugar])[0]

            input_data = np.array([[
                age, sex_encoded, bp_encoded, chol_encoded, bmi, sugar_encoded, na_to_k
            ]])

            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            drug = le_drug.inverse_transform(prediction)[0]

            advice_dict = {
                "DrugY": "Maintain a balanced diet, stay hydrated, and routine check-up recommended.",
                "drugA": "Monitor Blood Pressure closely. Restrict daily sodium intake and avoid heavy stress.",
                "drugB": "Follow a low-fat, fiber-rich diet to manage cholesterol alongside medication.",
                "drugC": "Keep a close check on electrolyte intake and follow up with a serum potassium test.",
                "drugX": "Continue standard dosage. Light aerobic exercise like walking is highly beneficial.",
                "Drug_Cardio": "Prioritize cardiovascular rest, limit processed salts, and monitor heart rate regularly."
            }
            
            current_advice = advice_dict.get(drug, "Follow standard clinical guidelines and physician dosage script.")

            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">🎯 AI Plan for {patient_name}</div>
                <div class="result-drug">💊 {drug}</div>
                <div class="result-advice">
                    <strong>💡 Clinical Advice:</strong><br>{current_advice}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.warning("""
            ⚠️ **Clinical Disclaimer:** This tool provides suggestions based on AI modeling and is for informational purposes only. 
            AI models can make mistakes. Please consult a qualified medical professional or doctor before initiating 
            any treatment or medication based on this report.
            """)
            
            current_date = datetime.date.today().strftime('%d-%B-%Y')
            report_content = f"""==================================================
              MEDIAI CLINICAL MEDICAL REPORT
==================================================
Date generated    : {current_date}
Patient Name      : {patient_name}
Age               : {age}
Gender            : {sex}
--------------------------------------------------
CLINICAL PARAMETERS SUMMARY:
--------------------------------------------------
Blood Pressure    : {bp}
Cholesterol       : {chol}
BMI               : {bmi}
Sugar Status      : {sugar}
Sodium-Potassium  : {na_to_k}
--------------------------------------------------
DIAGNOSTIC RECOMMENDATION:
--------------------------------------------------
RECOMMENDED DRUG  : {drug}

CLINICAL ADVICE:
{current_advice}
--------------------------------------------------
Disclaimer: Generated automatically by MediAI Assistant. 
Please consult a qualified doctor before taking medications.
=================================================="""

            st.download_button(
                label="📥 Download Clinical Report",
                data=report_content,
                file_name=f"Clinical_Report_{patient_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            
    st.markdown("</div>", unsafe_allow_html=True)

# COMPACT FOOTER
st.markdown("""
<div style='text-align:center; color:#64748b; font-size:11px; margin-top:20px;'>
MediAI Clinical Assistant • Secure Model Pipeline Active
</div>
""", unsafe_allow_html=True)