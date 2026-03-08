import streamlit as st
import joblib
import pandas as pd
import os
import sys

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hypertension Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header */
.hero-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    font-size: 1.05rem;
    color: #a0a8c0;
    margin-bottom: 2rem;
}

/* Glassmorphism card */
.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
}

.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7b83c4;
    margin-bottom: 1rem;
}

/* Result boxes */
.result-box {
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    margin-top: 0.5rem;
}
.result-normal   { background: linear-gradient(135deg, #1a472a, #27ae60); }
.result-elevated { background: linear-gradient(135deg, #7d6608, #f1c40f); color: #1a1a1a !important; }
.result-stage1   { background: linear-gradient(135deg, #784212, #e67e22); }
.result-stage2   { background: linear-gradient(135deg, #6e2121, #e74c3c); }
.result-severe   { background: linear-gradient(135deg, #4a0a0a, #c0392b); }

.result-stage {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.result-label {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.3rem;
}

/* BMI badge */
.bmi-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* Recommendation item */
.rec-item {
    background: rgba(255,255,255,0.06);
    border-left: 3px solid #7b83c4;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    color: #d0d6f0;
    font-size: 0.92rem;
}

/* Metric tiles */
.metric-tile {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value { font-size: 1.5rem; font-weight: 700; color: #fff; }
.metric-label { font-size: 0.75rem; color: #8890b0; margin-top: 0.2rem; }

/* Button override */
div.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* Slider label color */
label { color: #c0c8e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hypertension_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.recommend import recommend

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"❌ Model not found at `{MODEL_PATH}`. Run `python run_training.py` first.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ── Stage config ───────────────────────────────────────────────────────────────
STAGE_CONFIG = {
    "Normal":   {"emoji": "✅", "css": "result-normal",   "desc": "Your blood pressure is in a healthy range."},
    "Elevated": {"emoji": "⚠️", "css": "result-elevated", "desc": "Slightly elevated — lifestyle changes can help."},
    "Stage1":   {"emoji": "🔶", "css": "result-stage1",   "desc": "Stage 1 hypertension — monitor closely."},
    "Stage2":   {"emoji": "🔴", "css": "result-stage2",   "desc": "Stage 2 hypertension — clinical review advised."},
    "Severe":   {"emoji": "🚨", "css": "result-severe",   "desc": "Severe hypertension — seek medical attention."},
}

def bmi_category(bmi):
    if bmi < 18.5: return ("Underweight", "#5dade2")
    if bmi < 25:   return ("Normal",      "#27ae60")
    if bmi < 30:   return ("Overweight",  "#f39c12")
    return              ("Obese",         "#e74c3c")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🩺 Hypertension Stage Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Enter your health details to get an instant blood pressure stage assessment</p>', unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # Personal Info
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age    = st.slider("Age (years)", 18, 80, 35)
        height = st.slider("Height (cm)", 140, 200, 170)
    with c2:
        weight = st.slider("Weight (kg)", 40, 150, 70)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "👩 Female" if x == 1 else "👨 Male")
    st.markdown('</div>', unsafe_allow_html=True)

    # Blood Pressure
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">💉 Blood Pressure Readings</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        ap_hi = st.slider("Systolic (ap_hi) mmHg", 70, 260, 120)
    with c4:
        ap_lo = st.slider("Diastolic (ap_lo) mmHg", 40, 160, 80)
    st.markdown('</div>', unsafe_allow_html=True)

    # Lifestyle & Labs
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🧬 Clinical & Lifestyle Factors</p>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "High", 3: "Very High"}[x])
        smoke  = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c6:
        glucose = st.selectbox("Glucose", [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "High", 3: "Very High"}[x])
        alco   = st.selectbox("Alcohol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c7:
        cardio = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physically Active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍  Predict Hypertension Stage")

with right:
    bmi = weight / ((height / 100) ** 2)
    bmi_cat, bmi_color = bmi_category(bmi)

    # Live metrics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Your Health Snapshot</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-tile"><div class="metric-value">{bmi:.1f}</div><div class="metric-label">BMI</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-tile"><div class="metric-value">{ap_hi}/{ap_lo}</div><div class="metric-label">BP (mmHg)</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-tile"><div class="metric-value">{age}</div><div class="metric-label">Age</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;margin-top:0.8rem"><span class="bmi-badge" style="background:{bmi_color}22;color:{bmi_color};border:1px solid {bmi_color}55">BMI Category: {bmi_cat}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction result area
    if predict_btn:
        input_data = pd.DataFrame({
            "gender":      [gender],
            "height":      [height],
            "weight":      [weight],
            "ap_hi":       [ap_hi],
            "ap_lo":       [ap_lo],
            "cholesterol": [cholesterol],
            "gluc":        [glucose],
            "smoke":       [smoke],
            "alco":        [alco],
            "active":      [active],
            "cardio":      [cardio],
            "age_years":   [age],
            "bmi":         [bmi],
        })

        try:
            prediction = model.predict(input_data)[0]
            cfg = STAGE_CONFIG.get(prediction, STAGE_CONFIG["Normal"])

            st.markdown(f"""
            <div class="glass-card">
              <p class="section-header">🎯 Prediction Result</p>
              <div class="result-box {cfg['css']}">
                <p class="result-stage">{cfg['emoji']} {prediction}</p>
                <p class="result-label">{cfg['desc']}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Recommendations
            patient = {"smoke": smoke, "active": active}
            recs = recommend(prediction, patient)
            if recs:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-header">💡 Health Recommendations</p>', unsafe_allow_html=True)
                for r in recs:
                    st.markdown(f'<div class="rec-item">• {r}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem">🏥</div>
            <p style="color:#8890b0;font-size:1rem">Fill in your details on the left<br>and click <strong style="color:#a0a8c0">Predict</strong> to see your result.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;color:#5a6080;font-size:0.8rem">
    ⚕️ This tool is for informational purposes only and does not replace professional medical advice.
</div>
""", unsafe_allow_html=True)