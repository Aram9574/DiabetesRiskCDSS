# TFM: Clasificación de Enfermos Diabéticos Mediante Técnicas de ML

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Trabajo de Fin de Máster (TFM) — Inteligencia Artificial Aplicada a la Salud**
> *Autor: Alejandro Zakzuk* 
> *Herramienta de soporte a la decisión clínica para estratificación de riesgo de diabetes tipo 2 mediante modelos de Machine Learning.*

---

## 📌 Contexto Clínico y Objetivo del TFM

La diabetes tipo 2 es una de las enfermedades metabólicas de mayor prevalencia global, afectando a más de 537 millones de adultos. En el ámbito de atención primaria, diagnosticar tempranamente es fundamental para prevenir complicaciones severas y establecer medidas correctoras (modificación preventiva del estilo de vida o tratamientos tempranos).

Este proyecto es el producto de mi Trabajo de Fin de Máster y aborda la **estratificación del riesgo metabólico** creando un pipeline íntegro de Machine Learning: desde el preprocesamiento de datos clínicos con errores de medición, pasando por la validación cruzada estructurada, hasta la explicabilidad algorítmica de cada paciente utilizando herramientas matemáticas avanzadas (SHAP) y el despliegue de una interfaz clínica de apoyo.

🛑 **Aviso Legal:** *Esta herramienta no diagnostica diabetes*. Funciona exclusivamente bajo el paradigma de Soporte a la Decisión Clínica (Clinical Decision Support System o CDSS). Sugiere riesgos y complementa el juicio clínico, pero de ninguna forma lo reemplaza.

---

## 📊 Dataset: Pima Indians Diabetes Database

Se ha empleado como base inicial el histórico **Pima Indians Diabetes Dataset** originalmente recolectado por el National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). 

- **Población Base**: 768 mujeres de herencia nativo americana (Pima), mayores de 21 años.
- **Variable Objetivo (Outcome)**: Diagnóstico binario verificado de diabetes (0 = Ausencia, 1 = Presencia).
  
Las variables modeladas son las siguientes:

| Criterio Clínico | Relevancia y Significado Fisiológico |
|------------------|--------------------------------------|
| **Pregnancies** | Número de eventos gestacionales (Proxy de riesgo de diabetes gestacional). |
| **Glucose** | Niveles de glucemia plasmática (mg/dL) a las 2h tras tolerancia oral a la glucosa. |
| **BloodPressure** | Presión arterial diastólica (mmHg). |
| **SkinThickness** | Grosor del pliegue cutáneo tricipital (mm) — Proxy de la reserva de grasa subcutánea. |
| **Insulin** | Valores de insulina sérica originaria a las 2 horas (μU/mL). |
| **BMI** | Índice de Masa Corporal (kg/m²). |
| **DiabetesPedigreeFunction** | Puntuación asociada al riesgo genético y el historial clínico-familiar. |
| **Age** | Edad biológica en años. |

### Decisiones Críticas en el Tratamiento y Limpieza de Datos

En la fase inicial de correlación (EDA), se identificaron anomalías donde mediciones esenciales e incompatibles con la vida (como IMC o Glucosa sanguínea) exhibían un valor de "0". Fisiológicamente es un error de recolección común de aquella década. 

**Paso aplicado:** En lugar de purgar a las pacientes (lo cual hubiera mermado severamente el volumen de datos en casi un 49% en casos como la insulina), procedimos a la imputación cuidadosa de la **mediana agrupada por clase diagnosticada (diabéticas vs no diabéticas)** usando únicamente los valores de la partición segregada de entrenamiento (Train) para evitar la filtración de información cruzada ("Data Leakage" hacia la muestra de test).

---

## ⚙️ Arquitectura, Modelos y Experimentos

Al modelar el clasificador, realizamos en todo su proceso una búsqueda exhaustiva de hiperparámetros (GridSearchCV) y validación cruzada estratificada (StratifiedKFold) de 5 particiones en la capa de ajuste, confrontando los siguientes modelos:

1. **Regresión Logística**: Sirve de 'baseline genérico' lineal y control.
2. **Support Vector Machines (SVM con RBF)**: Capta fronteras fisiológicas curvas o no lineales del volumen de datos dimensional.
3. **Random Forest (RF)**: Se consagra analíticamente como el modelo definitivo de la investigación (Ensemble de Bagging), otorgando el mejor equilibrio entre rendimiento puro y capacidades de interpretabilidad.

### Métricas de Rendimiento Final en el Test Aislado (20%)

Para un sistema efectivo de cribado médico (screening), priorizamos localizar a todas las afectadas de manera fiable incluso si generamos un margen controlado de falsos positivos (actuación conservadora que se resuelve pidiendo una analítica confirmatoria ambulatoria de HbA1c a la paciente frente a ignorar un debut diabético peligroso). Por tanto, la optimización se estructuró evaluando basándose fundamentalmente en el **Área bajo la curva ROC (AUC) y la Sensibilidad**.

| Clasificador | Precisión (Acc) | Área Bajo la Curva (AUC-ROC) |
|--------------|-----------------|------------------------------|
| Regresión Logística | ~ 75.3% | 0.812 |
| SVM (Kernel RBF) | ~ 74.0% | 0.825 |
| **Random Forest** | **85.7%** | **0.942** |

*(Rendimientos derivados de la validación experimental y test aislado. Se reporta RF como iteración de despliegue)*

Además, realizamos un extenso **Análisis de Umbral Predictivo (Threshold)**, evaluando cómo rebajar el corte clasificador biomédico por defecto ($0.5$) hacia ($0.35$~$0.45$) permitiendo capturar de forma preventiva a muchos más verdaderos positivos en fases subclínicas previas de la enfermedad, al coste asumible de enviar interconsultas o confirmar con test al laboratorio de primaria.

---

## 🧠 Explicabilidad Clínica: Algoritmos SHAP Tool

Un modelo estricto de "caja negra" es inaceptable en el campo médico por parte de los clínicos. Para asegurar trazabilidad médica y confianza asistencial plena, se incorporó una capa sobre el árbol ensamblado de _SHapley Additive exPlanations_ (SHAP):
- **Importancia de Factores de Riesgo Global:** Destacamos nítidamente cómo las tres variables rectoras metabólicas consistentes empíricamente, **Glucosa**, **IMC (BMI)**, y **Edad** definieron los pesos de decisión primarios para el modelo.
- **Micro-Explicabilidad Local:** SHAP aporta también visualizaciones "waterfall plots" individualizadas, destilando frente de la consulta qué niveles biométricos o fenotipos particulares, en esta o la próxima paciente, se desvían de los óptimos y originan el porcentaje de riesgo particular otorgado al caso evaluado.

---

## 📂 Organización y Material del Repositorio

En este repositorio integramos los artefactos e interfaces desplegadas de mi TFM:
- `/src/TFM_Zakzuk_Codigo.py` : Scripts generados, depurados y extensamente comentados en español abordando pasos de imputación experta, validación cruzada y análisis SHAP Explainer. 
- `TFM_Alejandro_Zakzuk.pdf`: El entregable documental (Memoria Académica) completo del Trabajo de Fin de Máster.
- `app/` y `notebooks/`: Entornos analíticos secundarios y el diseño de la interfaz interactiva e iterativa construida bajo `Streamlit`.

---

## 🚀 Cómo Lanzar y Ver los Resultados o Plataforma

```bash
# 1. Clona el proyecto remotamente
git clone https://github.com/Aram9574/Pima_DiabetesRiskCDSS.git
cd Pima_DiabetesRiskCDSS

# 2. Instala las dependencias pertinentes en el ambiente virtual
pip install -r requirements.txt

# 3. Inicia la herramienta CDSS construida en Streamlit o lanza el análisis
streamlit run app/streamlit_app.py
```

---

*Cualquier recomendación clínica, "pull-request" técnico o reporte de incidencias en este repositorio será bienvenido y revisado a la mayor brevedad.*

**Alejandro Zakzuk** — Médico · IA Aplicada a la Sanidad (CEMP) · Salud Digital (UEM)
* [Conecta en LinkedIn](https://www.linkedin.com/in/alejandrozakzuk-ia-salud-digital/)
* [Sigue mi Portfolio GitHub](https://github.com/Aram9574)

