"""
Diabetes Risk CDSS — Streamlit Application (Full Clinical Suite v2.1)
Author: Alejandro Zakzuk | Physician · AI Applied to Health
---
Advanced CDSS with Predictive Interventions, Comprehensive ADA Guidelines, and Multi-Variate Simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import requests
from fpdf import FPDF
from datetime import datetime
from streamlit_lottie import st_lottie

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Risk CDSS | Clinical Suite 2.1",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Titles */
    .main-title { font-size: 2.8rem; font-weight: 800; color: #1e293b; letter-spacing: -0.04em; margin-bottom: 0.1rem; }
    .subtitle { font-size: 1.15rem; color: #64748b; margin-bottom: 2rem; font-weight: 400; }

    /* Risk Banners */
    .risk-banner {
        padding: 3.5rem 1.5rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s ease;
    }
    .risk-banner:hover { transform: scale(1.01); }
    .risk-high { background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%); color: white; }
    .risk-medium { background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); color: white; }
    .risk-low { background: linear-gradient(135deg, #10b981 0%, #064e3b 100%); color: white; }
    
    .risk-banner h3 { opacity: 0.9; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; font-size: 0.9rem; margin-bottom: 0.5rem; }
    .risk-banner h1 { font-size: 5rem; margin: 0; font-weight: 900; letter-spacing: -0.05em; line-height: 1; }
    .risk-banner p { opacity: 0.85; font-size: 1.1rem; font-weight: 500; margin-top: 0.5rem; }

    /* Professional Elements */
    .panel-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 20px;
        padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block; padding: 0.4rem 1rem; border-radius: 9999px;
        font-size: 0.8rem; font-weight: 700; margin-right: 0.6rem; border: 1px solid transparent;
    }
    .badge-blue { background: #eff6ff; color: #1d4ed8; border-color: #dbeafe; }
    .badge-slate { background: #f1f5f9; color: #475569; border-color: #e2e8f0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #f8fafc; border-radius: 12px 12px 0 0;
        padding: 0 24px; font-weight: 600; color: #64748b;
    }
    .stTabs [aria-selected="true"] { background-color: white !important; color: #1e293b !important; }

    /* Metric */
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 800 !important; color: #1e293b; }

    /* Sidebar QuickLoad */
    .quickload-btn { width: 100%; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

if 'pred_data' not in st.session_state: st.session_state.pred_data = None
if 'history' not in st.session_state: st.session_state.history = []

# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_resource
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    try:
        def get(f): return joblib.load(os.path.join(SRC_DIR, f))
        return get('model_rf.pkl'), get('shap_explainer.pkl'), get('scaler.pkl'), \
               get('imputation_stats.pkl'), get('cap_values.pkl'), get('feature_names.pkl')
    except: return None,None,None,None,None,None

def create_pdf_report(data, prob, label, insight):
    pdf = FPDF()
    pdf.add_page()
    def clean(t): return str(t).encode('latin-1', 'replace').decode('latin-1')
    
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 15, clean('DIABETES CLINICAL RISK ASSESSMENT'), 0, 1, 'C')
    pdf.set_font('Arial', '', 10); pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, clean(f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}'), 0, 1, 'C')
    pdf.ln(15)

    pdf.set_fill_color(248, 250, 252)
    pdf.rect(10, 45, 190, 45, 'F')
    pdf.set_font('Arial', 'B', 14); pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 12, clean(f'STRATIFICATION: {label}'), 0, 1, 'C')
    pdf.set_font('Arial', 'B', 32)
    pdf.cell(0, 20, f'{prob:.1%}', 0, 1, 'C')
    pdf.ln(20)

    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, clean('Patient Clinical Metrics:'), 0, 1)
    pdf.set_font('Arial', '', 10)
    for k, v in data.items():
        pdf.set_font('Arial', 'B', 10); pdf.cell(85, 8, clean(f'- {FEATURE_LABELS[k]}:'), 0, 0)
        pdf.set_font('Arial', '', 10); pdf.cell(100, 8, clean(f'{v} {REFERENCE_RANGES[k]["unit"]}'), 0, 1)
    
    pdf.ln(10); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, clean('AI Interpretation:'), 0, 1)
    pdf.set_font('Arial', 'I', 10); pdf.multi_cell(0, 6, clean(insight))
    
    pdf.ln(20); pdf.set_font('Arial', 'B', 9); pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 5, clean('Clinician: Alejandro Zakzuk, MD | Health Data Science'), 0, 1, 'C')
    pdf.set_font('Arial', '', 7); pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, clean('Verification: https://linkedin.com/in/Aram9574'), 0, 1, 'C')
    return bytes(pdf.output())

# ── Data Config ─────────────────────────────────────────────────────────────

REFERENCE_RANGES = {
    'Glucose': {'unit': 'mg/dL', 'min': 40, 'max': 400},
    'BMI': {'unit': 'kg/m²', 'min': 10, 'max': 70},
    'Age': {'unit': 'años', 'min': 21, 'max': 90},
    'DiabetesPedigreeFunction': {'unit': 'score', 'min': 0.0, 'max': 2.5},
    'Pregnancies': {'unit': 'n', 'min': 0, 'max': 17},
    'BloodPressure': {'unit': 'mmHg', 'min': 0, 'max': 122},
    'SkinThickness': {'unit': 'mm', 'min': 0, 'max': 99},
    'Insulin': {'unit': 'μU/mL', 'min': 0, 'max': 846},
}

FEATURE_LABELS = {
    'Glucose': 'Glucosa Plasma', 'BMI': 'IMC', 'Age': 'Edad',
    'DiabetesPedigreeFunction': 'Ant. Familiares', 'Pregnancies': 'Embarazos',
    'BloodPressure': 'Presion Diast.', 'SkinThickness': 'Pliegue Cutaneo', 'Insulin': 'Insulina 2h'
}

# ── Sidebar ─────────────────────────────────────────────────────────────────

model, explainer, scaler, imp_stats, cap_vals, feature_names = load_artifacts()

with st.sidebar:
    st.markdown("### 🧬 Registro Clinico")
    
    # QuickLoad Fix: Use buttons to set values and trigger rerun
    st.markdown("#### ⚡ QuickLoad Presets")
    col_q1, col_q2 = st.columns(2)
    if col_q1.button("🟢 Saludable", key="q_low"): 
        st.session_state.temp_in = {'Glucose': 88, 'BMI': 22.4, 'Age': 26, 'DiabetesPedigreeFunction': 0.2, 'Pregnancies': 0, 'BloodPressure': 72, 'SkinThickness': 20, 'Insulin': 85}
    if col_q2.button("🔴 Alto Riesgo", key="q_high"):
        st.session_state.temp_in = {'Glucose': 192, 'BMI': 36.5, 'Age': 54, 'DiabetesPedigreeFunction': 1.15, 'Pregnancies': 4, 'BloodPressure': 88, 'SkinThickness': 32, 'Insulin': 175}

    current_in = st.session_state.get('temp_in', {})
    
    in_vals = {}
    st.markdown("---")
    for f in ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
        v_min = float(REFERENCE_RANGES[f]['min'])
        v_max = float(REFERENCE_RANGES[f]['max'])
        raw_val = float(current_in.get(f, 100.0 if f == 'Glucose' else 25.0 if f == 'BMI' else 50.0))
        clamped_val = min(max(raw_val, v_min), v_max)
        in_vals[f] = st.number_input(FEATURE_LABELS[f], v_min, v_max, clamped_val, step=0.1 if f in ['BMI','DiabetesPedigreeFunction'] else 1.0)
    
    with st.expander("Laboratorio Secundario"):
        for f in ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin']:
            v_min = float(REFERENCE_RANGES[f]['min'])
            v_max = float(REFERENCE_RANGES[f]['max'])
            raw_val = float(current_in.get(f, 0.0))
            clamped_val = min(max(raw_val, v_min), v_max)
            in_vals[f] = st.number_input(FEATURE_LABELS[f], v_min, v_max, clamped_val)

    predict_btn = st.button("🔬 Realizar Analisis", use_container_width=True, type="primary")

# ── Logic ───────────────────────────────────────────────────────────────────

if predict_btn:
    df_raw = pd.DataFrame([in_vals])
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if df_raw[col].values[0] == 0: df_raw[col] = imp_stats[col]['overall_median']
    for col, caps in cap_vals.items(): 
        if col in df_raw.columns: df_raw[col] = df_raw[col].clip(caps['lower'], caps['upper'])
    
    patient_prep = df_raw[feature_names]
    prob = model.predict_proba(patient_prep)[0][1]
    
    label = "RIESGO ALTO" if prob >= 0.7 else "RIESGO MODERADO" if prob >= 0.4 else "RIESGO BAJO"
    color = "risk-high" if prob >= 0.7 else "risk-medium" if prob >= 0.4 else "risk-low"
    
    st.session_state.pred_data = {'prob': prob, 'label': label, 'color': color, 'prep': patient_prep, 'raw': in_vals}

# ── Main ───────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">🚀 Suite Clinica Digital</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Soporte a la Decision Medica con Inteligencia Artificial Explicable</p>', unsafe_allow_html=True)

if not st.session_state.pred_data:
    t1, t2 = st.tabs(["👋 Bienvenido", "📈 Insights Globales"])
    with t1:
        st.markdown("### ¿Como funciona?")
        st.write("Introduzca los datos del paciente en la barra lateral para generar una estratificacion de riesgo instantanea. Esta plataforma utiliza un modelo de bosque aleatorio (Random Forest) auditado para detectar patrones metabolicos invisibles al ojo humano.")
        st.markdown('<span class="badge badge-blue">ML Model Validated</span><span class="badge badge-slate">XAI Enabled</span><span class="badge badge-slate">ADA 2024 Compliant</span>', unsafe_allow_html=True)
        lottie_med = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_tutun9fe.json")
        if lottie_med: st_lottie(lottie_med, height=250)
    with t2: st.image("https://raw.githubusercontent.com/Aram9574/DiabetesRiskCDSS/main/notebooks/global_shap.png", caption="Pesos globales de decision")
else:
    d = st.session_state.pred_data
    tabs = st.tabs(["📊 Diagnostico y XAI", "⚖️ Simulador Multivariable", "📗 Guias ADA 2024"])
    
    with tabs[0]:
        st.markdown(f'<div class="risk-banner {d["color"]}"><h3>{d["label"]}</h3><h1>{d["prob"]:.1%}</h1><p>Probabilidad Estimada de Diabetes</p></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("#### 🧬 Explicacion de la Prediccion (SHAP)")
            shap_vals = explainer.shap_values(d['prep'])
            if isinstance(shap_vals, list): sv = shap_vals[1][0] if len(shap_vals)>1 else shap_vals[0][0]
            else: sv = shap_vals[0,:,1] if len(shap_vals.shape)==3 else shap_vals[0]
            sv = np.array(sv).flatten()
            
            sdf = pd.DataFrame({'Variable': [FEATURE_LABELS[f] for f in feature_names], 'SHAP': sv}).sort_values('SHAP')
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(sdf['Variable'], sdf['SHAP'], color=['#ef4444' if x > 0 else '#10b981' for x in sdf['SHAP']])
            ax.axvline(0, color='#e2e8f0', linestyle='--')
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig)
            
            top_f = [FEATURE_LABELS[f] for f in feature_names][np.argmax(np.abs(sv))]
            insight = f"El factor determinante es {top_f}. " + ("Este valor empuja el riesgo al alza." if sv[np.argmax(np.abs(sv))] > 0 else "Actua como factor protector.")
            st.info(f"💡 **Insight Clinico:** {insight}")
        with c2:
            st.markdown("#### 📋 Informe PDF")
            pdf_b = create_pdf_report(d['raw'], d['prob'], d['label'], insight)
            st.download_button("📥 Descargar Reporte Completo", pdf_b, f"Reporte_{datetime.now().strftime('%m%d')}.pdf", "application/pdf", use_container_width=True)
            st.dataframe(d['prep'].T.rename(columns={0: 'Valor'}), use_container_width=True)

    with tabs[1]:
        st.subheader("Simulador Predictivo de Intervencion")
        st.markdown("¿Que pasaria si modificamos multiples habitos de vida al mismo tiempo?")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            s_bmi = st.slider("Nuevo IMC", 15.0, 50.0, float(d['raw']['BMI']))
            s_age = st.slider("Edad (Proyeccion)", 21, 90, int(d['raw']['Age']))
        with col_s2:
            s_glu = st.slider("Nueva Glucosa", 60.0, 300.0, float(d['raw']['Glucose']))
            s_ins = st.slider("Nueva Insulina", 0.0, 500.0, float(d['raw']['Insulin']))
        
        sim_df = d['prep'].copy()
        sim_df['BMI'] = s_bmi; sim_df['Glucose'] = s_glu; sim_df['Age'] = s_age; sim_df['Insulin'] = s_ins
        p_sim = model.predict_proba(sim_df)[0][1]
        
        with col_s3:
            st.metric("Riesgo Simulado", f"{p_sim:.1%}", delta=f"{(p_sim - d['prob']):.1%}", delta_color="inverse")
            if p_sim < d['prob']: st.success("✅ La intervencion es efectiva.")
            else: st.warning("⚠️ No hay mejora significativa.")

    with tabs[2]:
        st.markdown("""
        ### Estandares de Cuidado ADA 2024
        
        #### 📈 Criterios Diagnosticos de Glucemia
        1. **Diabetes:** Glucosa en ayuno ≥ 126 mg/dL (7.0 mmol/L) o HbA1c ≥ 6.5%.
        2. **Prediabetes:** Glucosa en ayuno 100–125 mg/dL (5.6–6.9 mmol/L) o HbA1c 5.7–6.4%.
        3. **Prueba de Tolerancia (2h):** ≥ 200 mg/dL indica Diabetes.

        #### 📋 Recomendaciones Segun Estratificacion
        - **Riesgo Bajo (<40%):** Cribado rutinario cada 3 años (si >35 años).
        - **Riesgo Moderado (40-70%):** Vigilancia anual. Analitica inmediata si hay sintomas.
        - **Riesgo Alto (>70%):** Evaluación clínica proactiva. Cribado de complicaciones microvasculares.

        #### 🥗 Intervenciones del Estilo de Vida
        - **Peso:** Perdida de 7% del peso corporal inicial.
        - **Ejercicio:** 150 min/semana de actividad aerobica moderada a intensa.
        - **Nutricion:** Dieta basada en alimentos integrales y baja en azucares procesados.
        """)
        st.image("https://www.diabetes.org/sites/default/files/styles/default/public/2023-12/ADA-Logo.png", width=140)

st.markdown('<div class="disclaimer"><b>Aviso Legal:</b> Herramienta para uso administrativo y de portafolio. No reemplaza el diagnostico medico profesional.</div>', unsafe_allow_html=True)
