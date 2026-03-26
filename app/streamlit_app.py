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
    page_title="CDSS Riesgo de Diabetes",
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
    st.error(f"No se encontraron los artefactos del modelo. Ejecuta los notebooks 01–03 primero.\n\nError: {e}")

# ── Reference ranges for UI display ──────────────────────────────────────────

REFERENCE_RANGES = {
    'Glucose':                  {'unit': 'mg/dL', 'normal': '70–99 (ayuno)',   'min': 40,  'max': 400,  'step': 1},
    'BMI':                      {'unit': 'kg/m²', 'normal': '18.5–24.9',          'min': 10,  'max': 70,   'step': 0.1},
    'Age':                      {'unit': 'años',  'normal': '—',                  'min': 21,  'max': 90,   'step': 1},
    'DiabetesPedigreeFunction': {'unit': 'score',  'normal': '<0.5 (riesgo bajo)',    'min': 0.0, 'max': 2.5,  'step': 0.01},
    'Pregnancies':              {'unit': 'n',      'normal': '—',                  'min': 0,   'max': 17,   'step': 1},
    'BloodPressure':            {'unit': 'mmHg',   'normal': '60–80 (diastólica)', 'min': 0,  'max': 122,  'step': 1},
    'SkinThickness':            {'unit': 'mm',     'normal': '10–50',              'min': 0,   'max': 99,   'step': 1},
    'Insulin':                  {'unit': 'μU/mL',  'normal': '16–166 (2h post)',  'min': 0,   'max': 846,  'step': 1},
}

FEATURE_LABELS = {
    'Glucose':                  'Glucosa en Plasma',
    'BMI':                      'Índice de Masa Corporal (IMC)',
    'Age':                      'Edad',
    'DiabetesPedigreeFunction': 'Función de Pedigrí de Diabetes',
    'Pregnancies':              'Número de Embarazos',
    'BloodPressure':            'Presión Arterial Diastólica',
    'SkinThickness':            'Grosor del Pliegue Cutáneo',
    'Insulin':                  'Insulina Sérica (2h)',
}

# ── Header ────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-title">🩺 Predicción de Riesgo de Diabetes — CDSS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Herramienta de soporte a la decisión clínica · Random Forest · Pima Indians Diabetes Dataset</p>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.4rem; font-weight:700; color:#2d6a4f;">AUC-ROC 0.942</div>
        <div style="font-size:0.8rem; color:#666;">Rendimiento (Test Set)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if not artifacts_loaded:
    st.stop()

# ── Sidebar — Patient Input ───────────────────────────────────────────────────

st.sidebar.header("🔬 Parámetros del Paciente")
st.sidebar.markdown("Introduce los valores clínicos a continuación. Deja en 0 si no están disponibles (serán imputados).")
st.sidebar.markdown("---")

input_values = {}

priority_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
secondary_features = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin']

st.sidebar.markdown("**Predictores primarios**")
for feat in priority_features:
    ref = REFERENCE_RANGES[feat]
    val = st.sidebar.number_input(
        label=f"{FEATURE_LABELS[feat]} ({ref['unit']})",
        min_value=float(ref['min']),
        max_value=float(ref['max']),
        value=float((ref['min'] + ref['max']) / 2),
        step=float(ref['step']),
        help=f"Rango normal: {ref['normal']}"
    )
    input_values[feat] = val

st.sidebar.markdown("---")
st.sidebar.markdown("**Parámetros secundarios**")
for feat in secondary_features:
    ref = REFERENCE_RANGES[feat]
    val = st.sidebar.number_input(
        label=f"{FEATURE_LABELS[feat]} ({ref['unit']})",
        min_value=float(ref['min']),
        max_value=float(ref['max']),
        value=0.0,
        step=float(ref['step']),
        help=f"Rango normal: {ref['normal']} · 0 = no disponible"
    )
    input_values[feat] = val

predict_btn = st.sidebar.button("🔍 Predecir Riesgo", use_container_width=True, type="primary")

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
        risk_label = "RIESGO ALTO"
        risk_class = "risk-high"
        risk_emoji = "🔴"
    elif prob >= 0.4:
        risk_label = "RIESGO MODERADO"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_label = "RIESGO BAJO"
        risk_class = "risk-low"
        risk_emoji = "🟢"
    
    # Layout
    col_result, col_shap = st.columns([1, 1.5])
    
    with col_result:
        st.subheader("Resultado de la Predicción")
        st.markdown(f"""
        <div class="{risk_class}">
            <h2 style="margin:0;">{risk_emoji} {risk_label}</h2>
            <h1 style="margin:0.3rem 0; font-size:3rem;">{prob:.1%}</h1>
            <p style="margin:0; font-size:0.9rem;">Probabilidad estimada de diabetes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Valores utilizados (tras imputación)**")
        display_df = patient_ordered.T.copy()
        display_df.columns = ['Valor']
        # Use translated labels for index
        translated_index = [FEATURE_LABELS[f] for f in feature_names]
        display_df.index = translated_index
        display_df['Referencia'] = [REFERENCE_RANGES[f]['normal'] for f in feature_names]
        st.dataframe(display_df.round(2), use_container_width=True)
    
    with col_shap:
        st.subheader("Explicabilidad — ¿Qué impulsó esta predicción?")
        
        # 1. Get SHAP values
        shap_vals = explainer.shap_values(patient_ordered)
        
        # 2. Extract values for the 'Positive' class (Diabetes)
        # TreeExplainer often returns a list [neg_class_vals, pos_class_vals]
        if isinstance(shap_vals, list):
            sv_patient = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            # If it's a single array, handle cases where it might be (n_samples, n_features, n_classes)
            if len(shap_vals.shape) == 3:
                sv_patient = shap_vals[0, :, 1]
            else:
                sv_patient = shap_vals[0]
        
        # 3. Ensure 1D and correct length
        sv_patient = np.array(sv_patient).flatten()
        
        # Defensive check: feature alignment
        if len(sv_patient) != len(feature_names):
            # Fallback for some SHAP versions that might return a single value or different shape
            st.error(f"Error de alineación: {len(sv_patient)} valores SHAP para {len(feature_names)} características.")
            st.stop()
            
        # 4. Create DataFrame
        shap_df = pd.DataFrame({
            'Característica': [FEATURE_LABELS[f] for f in feature_names],
            'Valor SHAP': sv_patient,
            'Valor Paciente': patient_ordered.values[0]
        }).sort_values('Valor SHAP', key=abs, ascending=True)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#e53e3e' if v > 0 else '#38a169' for v in shap_df['Valor SHAP']]
        bars = ax.barh(shap_df['Característica'], shap_df['Valor SHAP'], color=colors, alpha=0.85, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xlabel('Valor SHAP (contribución al riesgo)', fontsize=10)
        ax.set_title('Contribuciones por Característica — Este Paciente', fontsize=11, fontweight='bold')
        
        for bar, (_, row) in zip(bars, shap_df.iterrows()):
            x_pos = row['Valor SHAP'] + (0.003 if row['Valor SHAP'] >= 0 else -0.003)
            ha = 'left' if row['Valor SHAP'] >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f"{row['Valor Paciente']:.1f}", va='center', ha=ha, fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **Cómo leer este gráfico:**
        - 🔴 Barras rojas: características que **aumentan** el riesgo.
        - 🟢 Barras verdes: características que **disminuyen** el riesgo.
        - Números en las barras: valor real del paciente para esa característica.
        """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Aviso Clínico:</strong> Esta herramienta tiene fines educativos y de investigación únicamente. 
    No está validada para uso clínico y no constituye consejo médico. 
    Las predicciones se basan en el dataset Pima Indians Diabetes (NIDDK) y pueden no generalizarse a otras poblaciones. 
    Cualquier decisión clínica requiere el juicio de un médico y el diagnóstico apropiado según guías clínicas (ADA 2024).
    </div>
    """, unsafe_allow_html=True)

else:
    # Default state — instructions
    st.info("👈 Introduce los parámetros del paciente en la barra lateral y haz clic en **Predecir Riesgo** para generar una evaluación.")
    
    # Show model info
    st.subheader("Sobre esta herramienta")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Modelo**  
        Random Forest Classifier  
        Optimizado con GridSearchCV  
        Validación cruzada estratificada (5-fold)
        """)
    with col2:
        st.markdown("""
        **Rendimiento (test set)**  
        AUC-ROC: 0.942  
        Accuracy: 85.7%  
        Brier score: < 0.15
        """)
    with col3:
        st.markdown("""
        **Explicabilidad**  
        SHAP TreeExplainer  
        Explicaciones globales y locales  
        Contribuciones por paciente
        """)
    
    st.markdown("---")
    st.markdown("""
    **Contexto clínico:** Esta herramienta implementa un pipeline de machine learning para la estratificación del riesgo de diabetes tipo 2 
    en un contexto de cribado de atención primaria. Está diseñada como una demostración de arquitectura de Sistema de Soporte a la Decisión Clínica (CDSS) 
    que combina modelos predictivos con explicabilidad — no como una herramienta clínica de producción.
    
    **Código fuente y metodología:** [GitHub](https://github.com/Aram9574/diabetes-risk-cdss)  
    **Autor:** [Alejandro Zakzuk](https://alejandrozakzuk.com) — Physician · AI Applied to Health (CEMP) · Digital Health (Universidad Europea de Madrid)
    """)
