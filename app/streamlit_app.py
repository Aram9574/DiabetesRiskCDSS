"""
Diabetes Risk CDSS — Streamlit Application (Full Clinical Suite)
Author: Alejandro Zakzuk | Physician · AI Applied to Health
---
Advanced CDSS with Explainable AI, PDF Reporting, and What-If Analysis.
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
    page_title="Diabetes Risk CDSS | Clinical Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (High-Performance Healthcare Dashboard) ───────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title { font-size: 2.2rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; }
    .subtitle { font-size: 1rem; color: #64748b; margin-bottom: 2rem; }

    /* Glass Panels */
    .st-emotion-cache-12w0qpk { background-color: #f8fafc; }
    
    .panel-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Risk Alerts */
    .risk-banner {
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        border: 2px solid;
    }
    .risk-high { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
    .risk-medium { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .risk-low { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }

    /* Professional badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background: #eff6ff; color: #1d4ed8; }
    .badge-slate { background: #f1f5f9; color: #475569; }

    /* Disclaimer box */
    .disclaimer {
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 2rem;
    }
    
    /* Interactive elements */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_resource
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

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
    
    # ── Header ──
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, 'INFORME CLINICO DE RIESGO METABOLICO (CDSS)', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Fecha de generacion: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(10)
    
    # ── Risk Section ──
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(10, 40, 190, 30, 'F')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'RESULTADO: {risk_label}', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 15, f'{prob:.1%}', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 5, 'Probabilidad estimada de presencia de diabetes tipo 2', 0, 1, 'C')
    pdf.ln(15)
    
    # ── Patient Data ──
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'Parametros Clinicos Analizados:', 0, 1)
    pdf.set_font('Arial', '', 10)
    for k, v in patient_data.items():
        label = FEATURE_LABELS.get(k, k)
        unit = REFERENCE_RANGES.get(k, {}).get('unit', '')
        pdf.cell(95, 8, f'- {label}:', 0, 0)
        pdf.cell(95, 8, f'{v} {unit}', 0, 1)
    
    pdf.ln(10)
    
    # ── AI Insight ──
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'Interpretacion de la Inteligencia Artificial:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, shap_desc)
    
    # ── Disclaimer ──
    pdf.ln(20)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 5, 'AVISO: Este informe es generado por un sistema experimental de IA (CDSS) para evaluacion de riesgo metabólico en el contexto de portafolio médico. No constituye diagnostico medico final. Se recomienda confirmación por laboratorio clínico (HbA1c/Glucosa Ayunas) segun estandares internacionales (ADA 2024).')
    
    return pdf.output(dest='S').encode('latin-1')

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

st.markdown('<h1 class="main-title">🚀 Suite Clínica Digital | Riesgo Diabetes</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Herramienta Avanzada de Estratificación Médica con IA Explicable</p>', unsafe_allow_html=True)

if model is None:
    st.error("🚨 Error de Configuración: Artefactos del modelo no detectados.")
    st.stop()

# ── Sidebar Workflow ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧬 Registro de Constantes")
    input_values = {}
    
    st.info("Variables Críticas (Inferencia Principal)")
    for f in ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
        input_values[f] = st.sidebar.number_input(
            f"{FEATURE_LABELS[f]} ({REFERENCE_RANGES[f]['unit']})",
            float(REFERENCE_RANGES[f]['min']), float(REFERENCE_RANGES[f]['max']),
            float((REFERENCE_RANGES[f]['min'] + REFERENCE_RANGES[f]['max'])/2)
        )
    
    with st.expander("Parámetros de Laboratorio Secundarios"):
        for f in ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin']:
            input_values[f] = st.number_input(
                f"{FEATURE_LABELS[f]} ({REFERENCE_RANGES[f]['unit']})",
                float(REFERENCE_RANGES[f]['min']), float(REFERENCE_RANGES[f]['max']), 0.0
            )

    st.markdown("---")
    predict_btn = st.button("🔬 Analizar Riesgo Metabólico", use_container_width=True, type="primary")

# ── Main Logic & Navigation ──────────────────────────────────────────────────

if not predict_btn:
    tab1, tab2 = st.tabs(["👋 Bienvenido", "📈 Inteligencia Poblacional"])
    
    with tab1:
        col_text, col_anim = st.columns([2, 1])
        with col_text:
            st.markdown("""
            ### Propuesta de Valor CDSS
            Esta plataforma integra un pipeline de **Machine Learning (Random Forest)** validado con el dataset de Pima Indians de la NIDDK. 
            
            **Características de Grado Médico:**
            - **Explicabilidad SHAP:** Cada predicción incluye un mapa de calor de las variables que influyen en el riesgo.
            - **Imputación Inteligente:** Manejo automático de datos faltantes mediante medianas clínicas.
            - **Cumplimiento ADA:** Referencias basadas en los estándares de cuidado vigentes.
            """)
            st.markdown('<span class="badge badge-blue">Python 3.10</span><span class="badge badge-slate">Random Forest</span><span class="badge badge-slate">SHAP Explainer</span>', unsafe_allow_html=True)
        
        with col_anim:
            lottie_medical = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3v62.json")
            if lottie_medical: st_lottie(lottie_medical, height=220)

    with tab2:
        st.subheader("Importancia Global de Variables")
        st.markdown("¿Cuáles son los factores que más mueven la balanza a nivel poblacional?")
        # Static representation of global importance if available
        st.image("https://raw.githubusercontent.com/Aram9574/DiabetesRiskCDSS/main/notebooks/global_shap.png", caption="Importancia Global (Datos de entrenamiento)")

else:
    # ── Inference Logic ──
    df_raw = pd.DataFrame([input_values])
    # Impute missing (0 case)
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if df_raw[col].values[0] == 0: df_raw[col] = imp_stats[col]['overall_median']
    # Clip
    for col, caps in cap_vals.items(): 
        if col in df_raw.columns: df_raw[col] = df_raw[col].clip(caps['lower'], caps['upper'])
    
    patient_prepared = df_raw[feature_names]
    prob = model.predict_proba(patient_prepared)[0][1]
    
    # UI Categories
    if prob >= 0.7: r_class, r_label = "risk-high", "RIESGO ALTO"
    elif prob >= 0.4: r_class, r_label = "risk-medium", "RIESGO MODERADO"
    else: r_class, r_label = "risk-low", "RIESGO BAJO"

    # ── Main Dashboard ──
    tab_res, tab_whatif, tab_guide = st.tabs(["📊 Diagnóstico de Riesgo", "⚖️ Análisis What-If", "📗 Guías Clínicas"])

    with tab_res:
        st.markdown(f'<div class="risk-banner {r_class}"><h3>{r_label}</h3><h1>{prob:.1%}</h1><p>Probabilidad Estimada de Diabetes</p></div>', unsafe_allow_html=True)
        st.ln(1)
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("#### Explicación IA de la Decisión")
            shap_vals = explainer.shap_values(patient_prepared)
            if isinstance(shap_vals, list): sv = shap_vals[1][0] if len(shap_vals)>1 else shap_vals[0][0]
            else: sv = shap_vals[0,:,1] if len(shap_vals.shape)==3 else shap_vals[0]
            sv = np.array(sv).flatten()
            
            # Simple bar plot
            s_df = pd.DataFrame({'Variable': [FEATURE_LABELS[f] for f in feature_names], 'SHAP': sv}).sort_values('SHAP')
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(s_df['Variable'], s_df['SHAP'], color=['#ef4444' if x > 0 else '#22c55e' for x in s_df['SHAP']])
            ax.axvline(0, color='black', alpha=0.3)
            st.pyplot(fig)
            
            # Logic for interpretation string
            top_feature = [FEATURE_LABELS[f] for f in feature_names][np.argmax(np.abs(sv))]
            clinical_insight = f"El factor determinante en este paciente es {top_feature}. " + ("Este valor está empujando el riesgo significativamente al alza." if sv[np.argmax(np.abs(sv))] > 0 else "Este valor actúa como factor protector en el perfil actual.")
            st.info(f"💡 **Insight IA:** {clinical_insight}")

        with c2:
            st.markdown("#### Informe del Paciente")
            st.dataframe(patient_prepared.T.rename(columns={0: 'Valor'}), use_container_width=True)
            
            # PDF Generation
            pdf_data = create_pdf_report(input_values, prob, r_label, clinical_insight)
            st.download_button(
                label="📥 Descargar Informe Clínico (PDF)",
                data=pdf_data,
                file_name=f"Informe_Diabetes_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with tab_whatif:
        st.subheader("Simulador Predictivo (What-If Analysis)")
        st.markdown("¿Cómo afectaría una intervención en el estilo de vida?")
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            bmi_sim = st.slider("Nuevo IMC deseado", 15.0, 50.0, float(input_values['BMI']))
            glucose_sim = st.slider("Nueva Glucosa deseada", 60.0, 300.0, float(input_values['Glucose']))
        
        # New inference for simulation
        sim_df = patient_prepared.copy()
        sim_df['BMI'] = bmi_sim
        sim_df['Glucose'] = glucose_sim
        prob_sim = model.predict_proba(sim_df)[0][1]
        
        with col_sim2:
            st.metric("Nueva Probabilidad", f"{prob_sim:.1%}", delta=f"{(prob_sim - prob):.1%}", delta_color="inverse")
            if prob_sim < prob: st.success("✅ La intervención simulada reduce significativamente el riesgo.")
            else: st.warning("⚠️ Los ajustes actuales no reducen el riesgo.")

    with tab_guide:
        st.markdown("""
        ### Estándares de Cuidado ADA 2024
        - **Criterio Diagnóstico:** Glucosa en ayuno ≥ 126 mg/dL o HbA1c ≥ 6.5%.
        - **Prediabetes:** Glucosa 100-125 mg/dL.
        - **Acción recomendada:** Si el riesgo es > 40%, se sugiere cribado analítico formal inmediato y evaluación de sindrome metabólico.
        """)
        st.image("https://www.diabetes.org/sites/default/files/styles/default/public/2023-12/ADA-Logo.png", width=150)

st.markdown('<div class="disclaimer"><b>Uso Experimental:</b> Esta herramienta es un proyecto de portafolio para demostrar capacidades en IA aplicada a Salud. No sustituye la consulta médica presencial.</div>', unsafe_allow_html=True)
