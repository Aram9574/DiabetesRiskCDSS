# Diabetes Risk Prediction — Clinical ML Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Herramienta de soporte a la decisión clínica para la estratificación del riesgo de diabetes tipo 2 en entornos de atención primaria.**
> Desarrollado como un proyecto de portfolio que demuestra un flujo completo de ML clínico: desde los datos brutos hasta la aplicación desplegada.

---

## Contexto Clínico

La diabetes tipo 2 afecta a más de 537 millones de adultos en todo el mundo (IDF, 2021), con una gran proporción de casos no diagnosticados en el momento de la consulta. La estratificación temprana del riesgo en atención primaria permite intervenciones dirigidas (modificación del estilo de vida, monitorización metabólica y derivación oportuna) antes de que se desarrolle una hiperglucemia manifiesta.

Este proyecto construye un pipeline de machine learning para apoyar esa decisión: a partir de los datos clínicos y antropométricos básicos de un paciente, se estima su probabilidad de diabetes y se visualizan las variables que impulsan dicho riesgo.

**Esta herramienta no diagnostica la diabetes.** Está diseñada como una capa de soporte para el clínico, bajo el paradigma de CDSS (*Clinical Decision Support System*): aumentar el juicio clínico, no reemplazarlo.

---

## Qué demuestra este proyecto

| Capa | Qué demuestra |
|---|---|
| Enfoque Clínico | Capacidad para traducir un problema clínico en una tarea de ML con definiciones de *outcomes* y selección de métricas adecuadas |
| Pipeline de ML | Flujo completo: EDA, preprocesamiento, comparación de modelos, ajuste de hiperparámetros y evaluación |
| Explicabilidad | Explicaciones locales y globales basadas en SHAP con interpretación clínica |
| Despliegue | App en Streamlit con UX informada clínicamente: rangos de referencia, categorías de riesgo y explicación por paciente |
| Comunicación | Documentación legible orientada tanto a audiencias clínicas como técnicas |

---

## Dataset

**Fuente:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — originalmente del National Institute of Diabetes and Digestive and Kidney Diseases.

**Población:** Mujeres adultas de herencia india Pima, edad ≥ 21 años.

**Variable objetivo:** Binaria — diagnóstico de diabetes confirmado por criterios de la OMS (glucosa en plasma ≥ 200 mg/dL o FPG ≥ 126 mg/dL en dos ocasiones).

**Características utilizadas:**

| Característica | Significado clínico |
|---|---|
| Pregnancies | Número de gestaciones (proxy de antecedentes de diabetes gestacional) |
| Glucose | Glucosa en plasma a las 2h en test de tolerancia oral (mg/dL) |
| BloodPressure | Presión arterial diastólica (mmHg) |
| SkinThickness | Grosor del pliegue cutáneo del tríceps (mm) — proxy de grasa subcutánea |
| Insulin | Insulina sérica a las 2 horas (μU/mL) |
| BMI | Índice de Masa Corporal (kg/m²) |
| DiabetesPedigreeFunction | Antecedentes familiares de diabetes — puntuación de riesgo genético |
| Age | Edad en años |

### Limitaciones Conocidas (Relevancia Clínica)

- **Muestra de sexo y etnia única.** La generalización a otras poblaciones es incierta. No se deben aplicar los umbrales derivados aquí en la práctica general sin validación previa.
- **Artefactos de valor cero.** Los ceros biológicamente implausibles en Glucosa, IMC, Presión Arterial, etc., representan datos faltantes, no ceros reales. Se manejaron explícitamente en el preprocesamiento (ver notebook 02).
- **Instantánea transversal.** No hay seguimiento longitudinal; no es posible distinguir entre diabetes incidente y prevalente.
- **Sin datos de medicación o estilo de vida.** Ausencia de variables confusoras como tratamiento antihiperglucemiante, actividad física o patrones dietéticos.

---

## Rendimiento del Modelo

| Modelo | AUC-ROC | Sensibilidad | Especificidad | Exactitud (Accuracy) |
|---|---|---|---|---|
| Regresión Logística | — | — | — | — |
| SVM (RBF) | — | — | — | — |
| **Random Forest** | **0.942** | — | — | **85.7%** |

> Evaluación completa en `notebooks/03_evaluation.ipynb`, incluyendo curvas de calibración, matrices de confusión y análisis de umbrales.

**¿Por qué AUC-ROC como métrica principal?**
En un contexto de cribado (screening), buscamos un modelo que identifique correctamente a los pacientes en riesgo (alta sensibilidad), incluso a costa de la especificidad. El AUC-ROC captura la capacidad de discriminación en todos los unbrales, lo que la convierte en la métrica principal adecuada. La exactitud por sí sola sería engañosa dada la prevalencia de la clase (~35%).

---

## Estructura del Proyecto

diabetes-risk-cdss/
├── data/
│   └── diabetes.csv              # Dataset original
├── notebooks/
│   ├── 01_eda.ipynb              # Análisis Exploratorio de Datos (EDA)
│   ├── 02_preprocessing.ipynb    # Ingeniería de variables e imputación
│   └── 03_evaluation.ipynb       # Comparación de modelos, SHAP, calibración
├── src/
│   ├── preprocessing.py          # Funciones de preprocesamiento reutilizables
│   ├── model.py                  # Entrenamiento y serialización del modelo
│   └── metrics.py                # Utilidades para métricas clínicas
├── app/
│   └── streamlit_app.py          # Calculadora de riesgo interactiva
├── requirements.txt
└── README.md


---

## Ejecución Local

```bash
git clone [https://github.com/Aram9574/diabetes-risk-cdss.git](https://github.com/Aram9574/diabetes-risk-cdss.git)
cd diabetes-risk-cdss
pip install -r requirements.txt

# Ejecutar notebooks
jupyter lab

# Ejecutar aplicación Streamlit
streamlit run app/streamlit_app.py
Demo en Vivo
Lanzar App ← enlace actualizado tras el despliegue

Descargo de Responsabilidad Clínica
Esta herramienta está destinada a fines educativos y de investigación. No está validada para uso clínico y no constituye asesoramiento médico. Cualquier aplicación en entornos clínicos reales requeriría una validación prospectiva, revisión regulatoria y supervisión institucional.

Autor
Alejandro Zakzuk — Médico · IA Aplicada a la Salud (CEMP) · Salud Digital (Universidad Europea de Madrid)

LinkedIn · Website · GitHub
