"""
Diabetes Risk CDSS — Streamlit Application
Author: Alejandro Zakzuk | Physician · AI Applied to Health
---
Clinical decision support tool for type 2 diabetes risk stratification.
Input: patient clinical parameters
Output: risk probability + SHAP-based explanation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Risk CDSS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .subtitle   { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .risk-high   { background: #fde8e8; border-left: 5px solid #e53e3e; padding: 1rem; border-radius: 6px; }
    .risk-medium { background: #fef3cd; border-left: 5px solid #d69e2e; padding: 1rem; border-radius: 6px; }
    .risk-low    { background: #e6f4ea; border-left: 5px solid #38a169; padding: 1rem; border-radius: 6px; }
    .disclaimer  { background: #f0f4ff; border: 1px solid #c3d0f0; padding: 0.8rem; border-radius: 6px; font-size: 0.85rem; color: #444; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 0.8rem 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    # BASE_DIR should point to the repository root
    # Since this file is in app/streamlit_app.py, the root is one level up
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    
    def get_path(filename):
        return os.path.join(SRC_DIR, filename)

    model     = joblib.load(get_path('model_rf.pkl'))
    explainer = joblib.load(get_path('shap_explainer.pkl'))
    scaler    = joblib.load(get_path('scaler.pkl'))
    imp_stats = joblib.load(get_path('imputation_stats.pkl'))
    cap_vals  = joblib.load(get_path('cap_values.pkl'))
    feat_names= joblib.load(get_path('feature_names.pkl'))
    
    return model, explainer, scaler, imp_stats, cap_vals, feat_names

try:
    model, explainer, scaler, imp_stats, cap_vals, feature_names = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Model artifacts not found. Run notebooks 01–03 first.\n\nError: {e}")

# ── Reference ranges for UI display ──────────────────────────────────────────

REFERENCE_RANGES = {
    'Glucose':                  {'unit': 'mg/dL', 'normal': '70–99 (fasting)',   'min': 40,  'max': 400,  'step': 1},
    'BMI':                      {'unit': 'kg/m²', 'normal': '18.5–24.9',          'min': 10,  'max': 70,   'step': 0.1},
    'Age':                      {'unit': 'years',  'normal': '—',                  'min': 21,  'max': 90,   'step': 1},
    'DiabetesPedigreeFunction': {'unit': 'score',  'normal': '<0.5 (low risk)',    'min': 0.0, 'max': 2.5,  'step': 0.01},
    'Pregnancies':              {'unit': 'n',      'normal': '—',                  'min': 0,   'max': 17,   'step': 1},
    'BloodPressure':            {'unit': 'mmHg',   'normal': '60–80 (diastolic)', 'min': 0,  'max': 122,  'step': 1},
    'SkinThickness':            {'unit': 'mm',     'normal': '10–50',              'min': 0,   'max': 99,   'step': 1},
    'Insulin':                  {'unit': 'μU/mL',  'normal': '16–166 (2h post)',  'min': 0,   'max': 846,  'step': 1},
}

FEATURE_LABELS = {
    'Glucose':                  'Plasma Glucose',
    'BMI':                      'Body Mass Index (BMI)',
    'Age':                      'Age',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree Function',
    'Pregnancies':              'Number of Pregnancies',
    'BloodPressure':            'Diastolic Blood Pressure',
    'SkinThickness':            'Triceps Skin Fold Thickness',
    'Insulin':                  '2h Serum Insulin',
}

# ── Header ────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-title">🩺 Diabetes Risk Prediction — CDSS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Clinical decision support tool · Random Forest · Pima Indians Diabetes Dataset</p>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.4rem; font-weight:700; color:#2d6a4f;">AUC-ROC 0.942</div>
        <div style="font-size:0.8rem; color:#666;">Test set performance</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if not artifacts_loaded:
    st.stop()

# ── Sidebar — Patient Input ───────────────────────────────────────────────────

st.sidebar.header("🔬 Patient Parameters")
st.sidebar.markdown("Enter clinical values below. Leave at 0 if not available (will be imputed).")
st.sidebar.markdown("---")

input_values = {}

priority_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
secondary_features = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin']

st.sidebar.markdown("**Primary predictors**")
for feat in priority_features:
    ref = REFERENCE_RANGES[feat]
    val = st.sidebar.number_input(
        label=f"{FEATURE_LABELS[feat]} ({ref['unit']})",
        min_value=float(ref['min']),
        max_value=float(ref['max']),
        value=float((ref['min'] + ref['max']) / 2),
        step=float(ref['step']),
        help=f"Normal range: {ref['normal']}"
    )
    input_values[feat] = val

st.sidebar.markdown("---")
st.sidebar.markdown("**Secondary parameters**")
for feat in secondary_features:
    ref = REFERENCE_RANGES[feat]
    val = st.sidebar.number_input(
        label=f"{FEATURE_LABELS[feat]} ({ref['unit']})",
        min_value=float(ref['min']),
        max_value=float(ref['max']),
        value=0.0,
        step=float(ref['step']),
        help=f"Normal range: {ref['normal']} · 0 = not available"
    )
    input_values[feat] = val

predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ── Preprocessing for inference ───────────────────────────────────────────────

def preprocess_input(input_values, imp_stats, cap_vals):
    df = pd.DataFrame([input_values])
    
    # Replace 0 with NaN for clinically implausible features, then impute
    implausible_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in implausible_zero:
        if df[col].values[0] == 0:
            if col in imp_stats:
                df[col] = imp_stats[col]['overall_median']
            else:
                df[col] = np.nan
    
    # Cap outliers
    for col, caps in cap_vals.items():
        if col in df.columns:
            df[col] = df[col].clip(caps['lower'], caps['upper'])
    
    return df

# ── Prediction & Display ──────────────────────────────────────────────────────

if predict_btn:
    # Preprocess
    patient_df = preprocess_input(input_values, imp_stats, cap_vals)
    patient_ordered = patient_df[feature_names]
    
    # Predict
    prob = model.predict_proba(patient_ordered)[0][1]
    pred = int(prob >= 0.5)
    
    # Risk category
    if prob >= 0.7:
        risk_label = "HIGH RISK"
        risk_class = "risk-high"
        risk_emoji = "🔴"
    elif prob >= 0.4:
        risk_label = "MODERATE RISK"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_label = "LOW RISK"
        risk_class = "risk-low"
        risk_emoji = "🟢"
    
    # Layout
    col_result, col_shap = st.columns([1, 1.5])
    
    with col_result:
        st.subheader("Prediction Result")
        st.markdown(f"""
        <div class="{risk_class}">
            <h2 style="margin:0;">{risk_emoji} {risk_label}</h2>
            <h1 style="margin:0.3rem 0; font-size:3rem;">{prob:.1%}</h1>
            <p style="margin:0; font-size:0.9rem;">Estimated probability of diabetes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Input values used (after imputation)**")
        display_df = patient_ordered.T.copy()
        display_df.columns = ['Value']
        display_df['Reference'] = [REFERENCE_RANGES[f]['normal'] for f in feature_names]
        st.dataframe(display_df.round(2), use_container_width=True)
    
    with col_shap:
        st.subheader("Explainability — What drove this prediction?")
        
        # SHAP local explanation
        shap_vals = explainer.shap_values(patient_ordered)
        if isinstance(shap_vals, list):
            sv_local = shap_vals[1][0]
        else:
            sv_local = shap_vals[0]
        
        # Waterfall-style bar chart
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': sv_local,
            'Patient Value': patient_ordered.values[0]
        }).sort_values('SHAP Value', key=abs, ascending=True)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#e53e3e' if v > 0 else '#38a169' for v in shap_df['SHAP Value']]
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, alpha=0.85, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xlabel('SHAP Value (contribution to diabetes risk)', fontsize=10)
        ax.set_title('Local Feature Contributions — This Patient', fontsize=11, fontweight='bold')
        
        for bar, (_, row) in zip(bars, shap_df.iterrows()):
            x_pos = row['SHAP Value'] + (0.003 if row['SHAP Value'] >= 0 else -0.003)
            ha = 'left' if row['SHAP Value'] >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f"{row['Patient Value']:.1f}", va='center', ha=ha, fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **How to read this chart:**
        - 🔴 Red bars: features pushing risk **higher**
        - 🟢 Green bars: features pushing risk **lower**  
        - Numbers on bars: patient's actual value for that feature
        """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Clinical Disclaimer:</strong> This tool is for educational and research purposes only. 
    It is not validated for clinical use and does not constitute medical advice. 
    Predictions are based on the Pima Indians Diabetes Dataset (NIDDK) and may not generalize to other populations. 
    Any clinical decision requires physician judgment and appropriate diagnostic workup per clinical guidelines (ADA 2024).
    </div>
    """, unsafe_allow_html=True)

else:
    # Default state — instructions
    st.info("👈 Enter patient parameters in the sidebar and click **Predict Risk** to generate an assessment.")
    
    # Show model info
    st.subheader("About this tool")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Model**  
        Random Forest Classifier  
        GridSearchCV optimized  
        5-fold stratified CV
        """)
    with col2:
        st.markdown("""
        **Performance (test set)**  
        AUC-ROC: 0.942  
        Accuracy: 85.7%  
        Brier score: < 0.15
        """)
    with col3:
        st.markdown("""
        **Explainability**  
        SHAP TreeExplainer  
        Global + local explanations  
        Per-patient feature contributions
        """)
    
    st.markdown("---")
    st.markdown("""
    **Clinical context:** This tool implements a machine learning pipeline for type 2 diabetes risk stratification 
    in a primary care screening context. It is designed as a demonstration of Clinical Decision Support System (CDSS) 
    architecture combining predictive modeling with explainability — not as a production clinical tool.
    
    **Source code & methodology:** [GitHub](https://github.com/Aram9574/diabetes-risk-cdss)  
    **Author:** [Alejandro Zakzuk](https://alejandrozakzuk.com) — Physician · AI Applied to Health (CEMP) · Digital Health (Universidad Europea de Madrid)
    """)
