"""
Diabetes Risk CDSS — Streamlit Application (Full Clinical Suite v2.0)
Author: Alejandro Zakzuk | Physician · AI Applied to Health
---
Enterprise-Grade CDSS with Explainable AI, PDF Reporting, Case Presets, and History Management.
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
    page_title="Diabetes Risk CDSS | Clinical Suite 2.0",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (Apple-Health Style Aesthetics) ────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Core Layout */
    .main-title { font-size: 2.5rem; font-weight: 800; color: #1e293b; letter-spacing: -0.04em; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.1rem; color: #64748b; margin-bottom: 2.5rem; font-weight: 400; }

    /* Glass Panels & Cards */
    .st-emotion-cache-12w0qpk { background-color: #f8fafc !important; }
    
    .panel-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    /* Risk Banners (Gradient Styles) */
    .risk-banner {
        padding: 3rem 1rem;
        border-radius: 20px;
        text-align: center;
        border: none;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .risk-high { background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%); color: white; }
    .risk-medium { background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); color: white; }
    .risk-low { background: linear-gradient(135deg, #10b981 0%, #064e3b 100%); color: white; }
    
    .risk-banner h3 { opacity: 0.9; font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .risk-banner h1 { font-size: 4rem; margin: 0; font-weight: 800; letter-spacing: -0.05em; }
    .risk-banner p { opacity: 0.8; font-size: 1rem; }

    /* Professional Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        border: 1px solid transparent;
    }
    .badge-blue { background: #eff6ff; color: #1d4ed8; border-color: #dbeafe; }
    .badge-slate { background: #f1f5f9; color: #475569; border-color: #e2e8f0; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Disclaimer box */
    .disclaimer {
        padding: 1.25rem;
        background: #f1f5f9;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        font-size: 0.9rem;
        color: #475569;
        margin-top: 3rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Management ──────────────────────────────────────────────────

if 'prediction' not in st.session_state: st.session_state.prediction = None
if 'history' not in st.session_state: st.session_state.history = []
if 'preset_values' not in st.session_state: st.session_state.preset_values = None

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
    def get_path(f): return os.path.join(SRC_DIR, f)
    try:
        return (joblib.load(get_path('model_rf.pkl')), 
                joblib.load(get_path('shap_explainer.pkl')),
                joblib.load(get_path('scaler.pkl')),
                joblib.load(get_path('imputation_stats.pkl')),
                joblib.load(get_path('cap_values.pkl')),
                joblib.load(get_path('feature_names.pkl')))
    except: return None,None,None,None,None,None

def create_pdf_report(patient_data, prob, risk_label, shap_desc):
    pdf = FPDF()
    pdf.add_page()
    def clean(t): return str(t).encode('latin-1', 'replace').decode('latin-1')

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 15, clean('DIABETES RISK CLINICAL REPORT (CDSS v2.0)'), 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, clean(f'Generated on: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'), 0, 1, 'C')
    pdf.ln(15)
    
    # Result Box
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(10, 45, 190, 40, 'F')
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, clean(f'RISK STRATIFICATION: {risk_label}'), 0, 1, 'C')
    pdf.set_font('Arial', 'B', 28)
    pdf.cell(0, 20, f'{prob:.1%}', 0, 1, 'C')
    pdf.ln(20)
    
    # Clinical Data
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, clean('Patient Clinical Profile:'), 0, 1)
    pdf.set_font('Arial', '', 10)
    for k, v in patient_data.items():
        label = FEATURE_LABELS.get(k, k)
        unit = REFERENCE_RANGES.get(k, {}).get('unit', '')
        pdf.set_font('Arial', 'B', 10); pdf.cell(90, 8, clean(f'- {label}:'), 0, 0)
        pdf.set_font('Arial', '', 10); pdf.cell(100, 8, clean(f'{v} {unit}'), 0, 1)
    
    pdf.ln(10)
    
    # Explainability
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, clean('Artificial Intelligence Interpretation:'), 0, 1)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 6, clean(shap_desc))
    
    # Footer
    pdf.ln(25)
    pdf.set_font('Arial', 'B', 8); pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 5, clean('Developed by: Alejandro Zakzuk | Physician & Data Scientist'), 0, 1, 'C')
    pdf.set_font('Arial', '', 7); pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, clean('LinkedIn: https://linkedin.com/in/Aram9574 | GitHub: https://github.com/Aram9574'), 0, 1, 'C')
    
    return bytes(pdf.output())

# ── Globals & Artifacts ──────────────────────────────────────────────────────

model, explainer, scaler, imp_stats, cap_vals, feature_names = load_artifacts()

REFERENCE_RANGES = {
    'Glucose': {'unit': 'mg/dL', 'normal': '70–99', 'min': 40, 'max': 400, 'step': 1},
    'BMI': {'unit': 'kg/m²', 'normal': '18.5–24.9', 'min': 10, 'max': 70, 'step': 0.1},
    'Age': {'unit': 'años', 'normal': '21-90', 'min': 21, 'max': 90, 'step': 1},
    'DiabetesPedigreeFunction': {'unit': 'score', 'normal': '<0.5', 'min': 0.0, 'max': 2.5, 'step': 0.01},
    'Pregnancies': {'unit': 'n', 'normal': '0-17', 'min': 0, 'max': 17, 'step': 1},
    'BloodPressure': {'unit': 'mmHg', 'normal': '60–80', 'min': 0, 'max': 122, 'step': 1},
    'SkinThickness': {'unit': 'mm', 'normal': '10–50', 'min': 0, 'max': 99, 'step': 1},
    'Insulin': {'unit': 'μU/mL', 'normal': '16–166', 'min': 0, 'max': 846, 'step': 1},
}

FEATURE_LABELS = {
    'Glucose': 'Glucosa Plasma', 'BMI': 'IMC', 'Age': 'Edad',
    'DiabetesPedigreeFunction': 'Ant. Familiares', 'Pregnancies': 'Embarazos',
    'BloodPressure': 'Presión Diast.', 'SkinThickness': 'Pliegue Cutáneo', 'Insulin': 'Insulina 2h'
}

# ── App Layout ───────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">🩺 CDSS v2.0 | Riesgo Diabetes</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estratificación Clínica mediante Inteligencia Artificial Explicable (XAI)</p>', unsafe_allow_html=True)

if model is None:
    st.error("🚨 Error de Configuración: Artefactos del modelo no detectados.")
    st.stop()

# ── Sidebar Workflow ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧬 Registro de Constantes")
    
    # Presets Section
    st.markdown("#### ⚡ Casos de Prueba (QuickLoad)")
    cp1, cp2 = st.columns(2)
    if cp1.button("🟢 Saludable"): st.session_state.preset_values = {'Glucose': 85, 'BMI': 22.5, 'Age': 28, 'DiabetesPedigreeFunction': 0.25, 'Pregnancies': 0, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 80}
    if cp2.button("🔴 Alto Riesgo"): st.session_state.preset_values = {'Glucose': 185, 'BMI': 38.2, 'Age': 58, 'DiabetesPedigreeFunction': 1.25, 'Pregnancies': 3, 'BloodPressure': 90, 'SkinThickness': 35, 'Insulin': 160}

    input_values = {}
    st.markdown("---")
    
    # Sidebar Input logic with Preset override
    for f in ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
        default_val = float(st.session_state.preset_values[f]) if st.session_state.preset_values else float((REFERENCE_RANGES[f]['min'] + REFERENCE_RANGES[f]['max'])/2)
        input_values[f] = st.number_input(f"{FEATURE_LABELS[f]} ({REFERENCE_RANGES[f]['unit']})", float(REFERENCE_RANGES[f]['min']), float(REFERENCE_RANGES[f]['max']), default_val, format="%.2f", key=f"sb_{f}")
    
    with st.expander("Parámetros Secundarios"):
        for f in ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin']:
            default_val = float(st.session_state.preset_values[f]) if st.session_state.preset_values else 0.0
            input_values[f] = st.number_input(f"{FEATURE_LABELS[f]} ({REFERENCE_RANGES[f]['unit']})", float(REFERENCE_RANGES[f]['min']), float(REFERENCE_RANGES[f]['max']), default_val, key=f"sb_{f}")

    st.markdown("---")
    predict_btn = st.button("🔬 Analizar Perfil Metabólico", use_container_width=True, type="primary")
    
    # History Memory
    if st.session_state.history:
        st.markdown("#### 🕒 Historial Reciente")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.caption(f"{h['time']} — {h['label']} ({h['prob']:.0%})")

# ── Main Logic ───────────────────────────────────────────────────────────────

if predict_btn:
    df_raw = pd.DataFrame([input_values])
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if df_raw[col].values[0] == 0: df_raw[col] = imp_stats[col]['overall_median']
    for col, caps in cap_vals.items(): 
        if col in df_raw.columns: df_raw[col] = df_raw[col].clip(caps['lower'], caps['upper'])
    
    patient_prepared = df_raw[feature_names]
    prob = model.predict_proba(patient_prepared)[0][1]
    
    if prob >= 0.7: r_class, r_label = "risk-high", "RIESGO ALTO"
    elif prob >= 0.4: r_class, r_label = "risk-medium", "RIESGO MODERADO"
    else: r_class, r_label = "risk-low", "RIESGO BAJO"

    st.session_state.prediction = {'prob': prob, 'r_class': r_class, 'r_label': r_label, 'patient_prepared': patient_prepared, 'input_values': input_values}
    st.session_state.history.append({'time': datetime.now().strftime("%H:%M"), 'label': r_label, 'prob': prob})

# Display logic
if st.session_state.prediction is None:
    tab1, tab2, tab_guide_init = st.tabs(["👋 Bienvenido", "📉 Estadística Global", "📗 Guías ADA"])
    with tab1:
        c_txt, c_anim = st.columns([1.5, 1])
        with c_txt:
            st.markdown("### El futuro del soporte a la decisión médica")
            st.write("Bienvenido a la Suite Clínica 2.0. Esta herramienta combina el poder del **Random Forest** con la transparencia de las técnicas **XAI (Explainable AI)** para proporcionar una evaluación de riesgo inmediata y accionable.")
            st.markdown("""
            ✅ **Validación Clínica:** Basado en dataset NIH-NIDDK.
            ⚡ **Respuesta Inmediata:** Inferencia en tiempo real (<0.1s).
            📜 **Informes Digitales:** Exportación estándar para historia clínica electrónica.
            """)
            st.markdown('<span class="badge badge-blue">ML Model: RF-114</span><span class="badge badge-slate">Accuracy: 84%</span><span class="badge badge-slate">Certified ADA Flow</span>', unsafe_allow_html=True)
        with c_anim:
            lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_tutun9fe.json")
            if lottie: st_lottie(lottie, height=250)
    with tab2: st.image("https://raw.githubusercontent.com/Aram9574/DiabetesRiskCDSS/main/notebooks/global_shap.png", caption="Arquitectura de pesos globales")
    with tab_guide_init: st.info("ℹ️ Realice una predicción para acceder a las guías detalladas.")
else:
    p = st.session_state.prediction
    tab_res, tab_whatif, tab_guide = st.tabs(["📊 Diagnóstico", "⚖️ Simulador What-If", "📗 Guías Clínicas"])

    with tab_res:
        st.markdown(f'<div class="risk-banner {p["r_class"]}"><h3>{p["r_label"]}</h3><h1>{p["prob"]:.1%}</h1><p>Probabilidad de Diabetes Tipo 2</p></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("#### 🧬 Explicabilidad del Riesgo (SHAP)")
            shap_vals = explainer.shap_values(p['patient_prepared'])
            if isinstance(shap_vals, list): sv = shap_vals[1][0] if len(shap_vals)>1 else shap_vals[0][0]
            else: sv = shap_vals[0,:,1] if len(shap_vals.shape)==3 else shap_vals[0]
            sv = np.array(sv).flatten()
            
            s_df = pd.DataFrame({'Variable': [FEATURE_LABELS[f] for f in feature_names], 'SHAP': sv}).sort_values('SHAP')
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(s_df['Variable'], s_df['SHAP'], color=['#ef4444' if x > 0 else '#10b981' for x in s_df['SHAP']])
            ax.axvline(0, color='#e2e8f0', alpha=0.9, linestyle='--')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title("Influencia de las variables en este paciente", fontsize=9, color='#64748b')
            st.pyplot(fig)
            
            top_feature = [FEATURE_LABELS[f] for f in feature_names][np.argmax(np.abs(sv))]
            insight = f"El factor de riesgo predominante es {top_feature}. " + ("Este indicador está elevando el riesgo significativamente." if sv[np.argmax(np.abs(sv))] > 0 else "Curiosamente, este valor compensa otros riesgos presentes.")
            st.info(f"💡 **Insight IA:** {insight}")

        with c2:
            st.markdown("#### 📋 Resumen Clínico")
            st.dataframe(p['patient_prepared'].T.rename(columns={0: 'Valor'}), use_container_width=True)
            
            pdf_data = create_pdf_report(p['input_values'], p['prob'], p['r_label'], insight)
            st.download_button("📥 Exportar Informe PDF Médico", data=pdf_data, file_name=f"Reporte_Clinico_{datetime.now().strftime('%d%m')}.pdf", mime="application/pdf", use_container_width=True)

    with tab_whatif:
        st.subheader("Simulador de Intervención")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            bmi_sim = st.slider("IMC Meta", 10.0, 50.0, float(p['input_values']['BMI']))
            glu_sim = st.slider("Glucosa Meta", 50.0, 300.0, float(p['input_values']['Glucose']))
        sim_df = p['patient_prepared'].copy()
        sim_df['BMI'] = bmi_sim; sim_df['Glucose'] = glu_sim
        prob_sim = model.predict_proba(sim_df)[0][1]
        with col_s2:
            st.metric("Nueva Probabilidad", f"{prob_sim:.1%}", delta=f"{(prob_sim - p['prob']):.1%}", delta_color="inverse")
            if prob_sim < p['prob']: st.success("✅ La reducción en parámetros metabólicos reduce significativamente el riesgo.")

    with tab_guide:
        st.markdown("### Guías ADA 2024 & Estrategias")
        st.markdown("""
        | Diagnóstico | Glucosa (mg/dL) | Acción suguerida |
        | :--- | :---: | :--- |
        | **Normal** | < 100 | Prevención y Estilo de Vida |
        | **Prediabetes** | 100 - 125 | Monitorización Semestral |
        | **Diabetes** | ≥ 126 | Evaluación Clínica Urgente |
        """)
        st.image("https://www.diabetes.org/sites/default/files/styles/default/public/2023-12/ADA-Logo.png", width=120)

st.markdown('<div class="disclaimer"><b>Nota Médica Legal:</b> Este sistema es un CDSS (Clinical Decision Support System) de portafolio profesional. Los resultados son estimaciones probabilísticas y deben ser validados por un médico colegiado bajo el contexto clínico individual de cada paciente.</div>', unsafe_allow_html=True)
