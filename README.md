# Predicción de Riesgo de Diabetes — Portfolio de ML Clínico

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Herramienta de soporte a la decisión clínica para estratificación de riesgo de diabetes tipo 2 en atención primaria.**
> Proyecto de portfolio que demuestra un pipeline completo de ML clínico: desde los datos brutos hasta la aplicación desplegada.

---

## Contexto Clínico

La diabetes tipo 2 afecta a más de 537 millones de adultos en todo el mundo (FID, 2021), y una proporción importante permanece sin diagnóstico en el momento de la consulta. La estratificación temprana del riesgo en atención primaria permite intervenciones dirigidas — modificación del estilo de vida, seguimiento metabólico y derivación oportuna — antes de que se desarrolle hiperglucemia franca.

Este proyecto construye un pipeline de aprendizaje automático para apoyar esa decisión: a partir de datos clínicos y antropométricos básicos del paciente, estima la probabilidad de diabetes e identifica las variables que determinan ese riesgo.

**Esta herramienta no diagnostica diabetes.** Está diseñada como una capa de soporte a la decisión clínica, coherente con el paradigma CDSS (Clinical Decision Support System) — complementa el juicio clínico, no lo reemplaza.

---

## Qué Demuestra Este Proyecto

| Capa | Qué evidencia |
|---|---|
| Encuadre clínico | Capacidad de traducir un problema clínico en una tarea de ML con definición adecuada del desenlace y selección de métricas |
| Pipeline de ML | Flujo completo: EDA, preprocesamiento, comparación de modelos, ajuste de hiperparámetros, evaluación |
| Explicabilidad | Explicaciones locales y globales basadas en SHAP con interpretación clínica |
| Despliegue | Aplicación Streamlit con UX clínicamente informada: rangos de referencia, categorías de riesgo, explicación por paciente |
| Comunicación | Documentación legible orientada a audiencias clínicas y técnicas |

---

## Dataset

**Fuente:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — originalmente del National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

**Población:** Mujeres adultas de herencia Pima, edad ≥ 21 años.

**Variable objetivo:** Binaria — diagnóstico de diabetes confirmado por criterios OMS (glucemia plasmática ≥ 200 mg/dL o GPB ≥ 126 mg/dL en dos ocasiones).

**Variables utilizadas:**

| Variable | Significado clínico |
|---|---|
| Pregnancies | Número de eventos gestacionales (proxy de historia de diabetes gestacional) |
| Glucose | Glucemia plasmática a las 2h en prueba de tolerancia oral a la glucosa (mg/dL) |
| BloodPressure | Presión arterial diastólica (mmHg) |
| SkinThickness | Grosor del pliegue cutáneo tricipital (mm) — proxy de grasa subcutánea |
| Insulin | Insulina sérica a las 2 horas (μU/mL) |
| BMI | Índice de Masa Corporal (kg/m²) |
| DiabetesPedigreeFunction | Historia familiar de diabetes — puntuación de riesgo genético |
| Age | Edad en años |

### Limitaciones Conocidas (Relevancia Clínica)

- **Muestra de un único sexo y una única etnia.** La generalizabilidad a otras poblaciones es incierta. No aplicar los umbrales derivados aquí a la práctica general sin validación previa.
- **Artefactos de valor cero.** Los ceros biológicamente implausibles en Glucose, BMI, BloodPressure, etc. representan datos faltantes, no valores reales. Se tratan explícitamente en el preprocesamiento (ver notebook 02).
- **Corte transversal.** Sin seguimiento longitudinal; no es posible distinguir diabetes incidente de diabetes prevalente.
- **Sin datos de medicación ni estilo de vida.** Ausentes factores de confusión como tratamiento antihiperglucemiante, actividad física o patrones dietéticos.

---

## Rendimiento del Modelo

| Modelo | AUC-ROC | Sensibilidad | Especificidad | Exactitud |
|---|---|---|---|---|
| Regresión Logística | — | — | — | — |
| SVM (RBF) | — | — | — | — |
| **Random Forest** | **0.942** | — | — | **85.7%** |

> Evaluación completa en `notebooks/03_evaluation.ipynb`, incluyendo curvas de calibración, matrices de confusión y análisis de umbral.

**¿Por qué AUC-ROC como métrica principal?**
En un contexto de cribado, se busca un modelo que identifique correctamente a los pacientes en riesgo (alta sensibilidad), aunque sea a costa de algo de especificidad. El AUC-ROC captura la capacidad de discriminación en todos los umbrales, lo que lo convierte en la métrica principal apropiada. La exactitud por sí sola sería engañosa dado el desbalance de clases (~35% de tasa positiva).

---

## Estructura del Proyecto
```
diabetes-risk-cdss/
├── data/
│   └── diabetes.csv              # Dataset original
├── notebooks/
│   ├── 01_eda.ipynb              # Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb    # Ingeniería de variables e imputación
│   └── 03_evaluation.ipynb       # Comparación de modelos, SHAP, calibración
├── src/
│   ├── preprocessing.py          # Funciones de preprocesamiento reutilizables
│   ├── model.py                  # Entrenamiento y serialización del modelo
│   └── metrics.py                # Utilidades de métricas clínicas
├── app/
│   └── streamlit_app.py          # Calculadora de riesgo interactiva
├── requirements.txt
└── README.md
```

---

## Ejecución Local
```bash
git clone https://github.com/Aram9574/diabetes-risk-cdss.git
cd diabetes-risk-cdss
pip install -r requirements.txt

# Ejecutar notebooks
jupyter lab

# Ejecutar aplicación Streamlit
streamlit run app/streamlit_app.py
```

---

## Demo en Vivo

[Abrir aplicación](https://your-streamlit-url.streamlit.app) ← *enlace actualizado tras el despliegue*

---

## Aviso Legal Clínico

Esta herramienta está destinada a fines educativos y de investigación. No está validada para uso clínico y no constituye consejo médico. Cualquier aplicación en entornos clínicos reales requeriría validación prospectiva, revisión regulatoria y supervisión institucional.

---

## Autor

**Alejandro Zakzuk** — Médico · IA Aplicada a la Sanidad (CEMP) · Salud Digital (Universidad Europea de Madrid)

[LinkedIn](https://linkedin.com/in/alejandrozakzuk-ia-salud-digital) · [Sitio web](https://alejandrozakzuk.com) · [GitHub](https://github.com/Aram9574)
