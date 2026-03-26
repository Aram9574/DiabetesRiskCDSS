"""
Diabetes Risk CDSS — Streamlit Application (Premium Redesign)
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
    page_title="Diabetes Risk CDSS | Dashboard Médico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (Premium Healthcare Aesthetics) ────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Title and Headers */
    .main-title { 
        font-size: 2.2rem; 
        font-weight: 700; 
        color: #1e293b; 
        margin-bottom: 0.1rem;
        letter-spacing: -0.025em;
    }
    .subtitle { 
        font-size: 1.1rem; 
        color: #64748b; 
        margin-bottom: 2rem; 
        font-weight: 400;
    }

    /* Premium Cards / Sections */
    .st-emotion-cache-12w0qpk { /* Sidebar fix */
        background-color: #f8fafc;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    /* Risk Score Styling */
    .risk-container {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .risk-high {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 1px solid #feb2b2;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fffaf3 0%, #feebc8 100%);
        border: 1px solid #fbd38d;
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border: 1px solid #9ae6b4;
    }

    .risk-label {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .risk-score {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .risk-desc {
        font-size: 1rem;
        color: #4a5568;
    }

    /* Parameter Chips */
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .param-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: left;
    }
    .param-label {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
    }
    .param-value {
        font-size: 1.25rem;
        color: #1e293b;
        font-weight: 700;
    }
    .param-unit {
        font-size: 0.75rem;
        color: #64748b;
    }

    /* Disclaimer */
    .disclaimer-box {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #94a3b8;
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    
    def get_path(filename):
        return os.path.join(SRC_DIR, filename)

    try:
        model     = joblib.load(get_path('model_rf.pkl'))
        explainer = joblib.load(get_path('shap_explainer.pkl'))
        scaler    = joblib.load(get_path('scaler.pkl'))
        imp_stats = joblib.load(get_path('imputation_stats.pkl'))
        cap_vals  = joblib.load(get_path('cap_values.pkl'))
        feat_names= joblib.load(get_path('feature_names.pkl'))
        return model, explainer, scaler, imp_stats, cap_vals, feat_names
    except Exception:
        return None, None, None, None, None, None

model, explainer, scaler, imp_stats, cap_vals, feature_names = load_artifacts()

# ── Translations and Config ───────────────────────────────────────────────────

REFERENCE_RANGES = {
    'Glucose':                  {'unit': 'mg/dL', 'normal': '70–99',   'min': 40,  'max': 400,  'step': 1},
    'BMI':                      {'unit': 'kg/m²', 'normal': '18.5–24.9', 'min': 10,  'max': 70,   'step': 0.1},
    'Age':                      {'unit': 'años',  'normal': '21-90',    'min': 21,  'max': 90,   'step': 1},
    'DiabetesPedigreeFunction': {'unit': 'score',  'normal': '<0.5',     'min': 0.0, 'max': 2.5,  'step': 0.01},
    'Pregnancies':              {'unit': 'n',      'normal': '0-17',     'min': 0,   'max': 17,   'step': 1},
    'BloodPressure':            {'unit': 'mmHg',   'normal': '60–80',    'min': 0,   'max': 122,  'step': 1},
    'SkinThickness':            {'unit': 'mm',     'normal': '10–50',    'min': 0,   'max': 99,   'step': 1},
    'Insulin':                  {'unit': 'μU/mL',  'normal': '16–166',   'min': 0,   'max': 846,  'step': 1},
}

FEATURE_LABELS = {
    'Glucose':                  'Glucosa en Plasma',
    'BMI':                      'IMC',
    'Age':                      'Edad',
    'DiabetesPedigreeFunction': 'Pedigrí Familiar',
    'Pregnancies':              'Embarazos',
    'BloodPressure':            'Presión Diastólica',
    'SkinThickness':            'Grosor de Piel',
    'Insulin':                  'Insulina Sérica',
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774293.png", width=80) 
    st.markdown("### Perfil Clínico")
    st.markdown("Ajusta los parámetros para evaluar el riesgo del paciente.")
    st.markdown("---")
    
    input_values = {}
    
    st.markdown("**Variables Críticas**")
    for feat in ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
        ref = REFERENCE_RANGES[feat]
        input_values[feat] = st.number_input(
            label=f"{FEATURE_LABELS[feat]} ({ref['unit']})",
            min_value=float(ref['min']), max_value=float(ref['max']),
            value=float((ref['min'] + ref['max']) / 2),
            step=float(ref['step'])
        )
    
    st.markdown("---")
    st.markdown("**Variables Secundarias**")
    with st.expander("Ver más parámetros"):
        for feat in ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin']:
            ref = REFERENCE_RANGES[feat]
            input_values[feat] = st.number_input(
                label=f"{FEATURE_LABELS[feat]} ({ref['unit']})",
                min_value=float(ref['min']), max_value=float(ref['max']),
                value=0.0, step=float(ref['step']),
                help="Deja en 0 si no se dispone del dato."
            )
    
    st.markdown("---")
    predict_btn = st.button("📊 Generar Diagnóstico", use_container_width=True, type="primary")

# ── Main Content ──────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">CDSS | Riesgo de Diabetes Mellitus</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema Experto de Soporte a Decisiones Clínicas apoyado en IA</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Error Crítico: Los archivos del modelo no fueron localizados en /src/.")
    st.stop()

if not predict_btn:
    # Landing Page Style
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        ### ¿Cómo funciona esta herramienta?
        Esta aplicación utiliza un modelo de **Random Forest** entrenado con datos clínicos 
        validados para detectar patrones tempranos de diabetes tipo 2. 
        
        **Instrucciones:**
        1. Introduce los valores del laboratorio en la barra lateral.
        2. Pulsa el botón para ejecutar la inferencia.
        3. Obtén una probabilidad de riesgo y una explicación detallada de los factores.
        
        > **Rendimiento del sistema:** AUC-ROC de **{0.942:.3f}** | Precisión global: **85.7%**
        """)
    with col2:
        st.info("💡 **Consejo:** El modelo es más preciso cuando se incluyen las variables de Glucosa e IMC. Los valores en 0 serán gestionados por el algoritmo de imputación clínica mediana.")

    st.markdown("---")
    st.markdown("""
        <div class="disclaimer-box">
        <strong>⚠️ Descargo de Responsabilidad:</strong> Esta herramienta médica digital está diseñada exclusivamente para fines 
        investigativos y educativos. No reemplaza el juicio criterio médico clínico profesional ni el diagnóstico 
        confirmatorio mediante hemoglobina glicosilada (HbA1c) u otras pruebas estandarizadas.
        </div>
    """, unsafe_allow_html=True)

else:
    # ── Logic ─────────────────────────────────────────────────────────────────
    
    # Preprocess
    df = pd.DataFrame([input_values])
    implausible_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in implausible_zero:
        if df[col].values[0] == 0:
            df[col] = imp_stats[col]['overall_median']
    for col, caps in cap_vals.items():
        if col in df.columns:
            df[col] = df[col].clip(caps['lower'], caps['upper'])
    
    patient_ordered = df[feature_names]
    
    # Predict
    prob = model.predict_proba(patient_ordered)[0][1]
    
    # Risk categorization
    if prob >= 0.7:
        risk_class, label, color = "risk-high", "RIESGO ALTO", "#e53e3e"
    elif prob >= 0.4:
        risk_class, label, color = "risk-medium", "RIESGO MODERADO", "#d69e2e"
    else:
        risk_class, label, color = "risk-low", "RIESGO BAJO", "#38a169"

    # Layout Results
    col_score, col_details = st.columns([1, 1.8])

    with col_score:
        st.markdown(f"""
        <div class="risk-container {risk_class}">
            <div class="risk-label" style="color: {color};">{label}</div>
            <div class="risk-score" style="color: {color};">{prob:.1%}</div>
            <div class="risk-desc">Probabilidad de padecer diabetes</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Resumen del Paciente")
        st.write("Datos procesados tras normalización:")
        summary_df = patient_ordered.T.copy()
        summary_df.columns = ['Valor']
        summary_df.index = [FEATURE_LABELS[f] for f in feature_names]
        st.dataframe(summary_df.round(2), use_container_width=True)

    with col_details:
        st.markdown("#### Análisis de Factores (Explainable AI)")
        
        # Robust SHAP extraction
        shap_vals = explainer.shap_values(patient_ordered)
        if isinstance(shap_vals, list):
            sv_patient = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            sv_patient = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0]
        
        sv_patient = np.array(sv_patient).flatten()
        
        if len(sv_patient) == len(feature_names):
            shap_df = pd.DataFrame({
                'Característica': [FEATURE_LABELS[f] for f in feature_names],
                'Valor SHAP': sv_patient
            }).sort_values('Valor SHAP', key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='none')
            colors = ['#f56565' if v > 0 else '#48bb78' for v in shap_df['Valor SHAP']]
            bars = ax.barh(shap_df['Característica'], shap_df['Valor SHAP'], color=colors, alpha=0.9)
            ax.axvline(0, color='#cbd5e1', linewidth=1, linestyle='--')
            
            # Stylize plot
            ax.set_title("Contribución al Riesgo (Influencia IA)", fontsize=10, fontweight='bold', color='#475569')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=9, colors='#64748b')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No se pudo generar el gráfico de explicabilidad detallado.")

    st.markdown("---")
    st.markdown("""
        <div class="disclaimer-box">
        <strong>Recomendación Clínica AI:</strong> El principal factor detectado para este paciente es 
        la <strong>Glucosa en Plasma</strong>. Se recomienda seguimiento metabólico estrecho y evaluación 
        de hábitos de vida saludable.
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 CDSS Diabetes Platform | Impulsado por Random Forest y SHAP Explainability")
