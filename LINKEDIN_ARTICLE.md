# Construí una herramienta de soporte a la decisión clínica para diabetes tipo 2 — esto es lo que aprendí

Llevo un año trabajando en la intersección entre medicina clínica e inteligencia artificial. Y hay una tensión que nadie te explica bien: la distancia entre un modelo que funciona y una herramienta que un clínico querría usar de verdad.

Este proyecto fue mi intento de reducir esa distancia.

---

## El problema clínico

La diabetes tipo 2 afecta a más de 537 millones de personas en el mundo. Una proporción importante está sin diagnóstico en el momento de la primera consulta. El cribado sistemático en atención primaria existe, pero el tiempo clínico disponible para evaluar riesgo en pacientes asintomáticos es limitado.

Un modelo que estratifique riesgo con datos básicos — glucosa, IMC, edad, antecedentes familiares — podría ayudar a priorizar quién necesita una PTOG o seguimiento más estrecho. Esa es la hipótesis clínica detrás de este proyecto.

---

## Lo que construí

Un pipeline de Machine Learning end-to-end con tres componentes:

**1. Pipeline de datos con criterio clínico**

El dataset Pima Indians tiene un problema que muchos tutoriales ignoran: los ceros en glucosa, IMC o tensión arterial no son ceros reales. Son datos ausentes codificados como cero. Un BMI de 0 es fisiológicamente imposible.

Resolverlo bien importa porque la estrategia de imputación cambia el modelo. Usé imputación por mediana estratificada por clase — la mediana de glucosa en pacientes diabéticos es diferente a la de no diabéticos, y mezclarlas introduce sesgo.

**2. Comparativa de modelos con métricas clínicas**

Comparé Logistic Regression, SVM con kernel RBF, y Random Forest. Elegí AUC-ROC como métrica principal, no accuracy.

¿Por qué? En cribado, el coste de un falso negativo (no detectar a alguien con diabetes) es mayor que el de un falso positivo (citar a alguien que no la tiene). El AUC-ROC captura la capacidad discriminativa del modelo en todos los umbrales, lo que permite elegir el punto de operación clínicamente apropiado.

Random Forest con GridSearchCV ganó: AUC-ROC 0.83 en validación cruzada 5-fold.

**3. Explicabilidad con SHAP**

Esta fue la parte más importante del proyecto.

SHAP (SHapley Additive exPlanations) permite saber exactamente qué variables empujaron la predicción en un paciente concreto. No es suficiente decirle a un médico "este paciente tiene 74% de riesgo". Necesita saber: ¿es por la glucosa? ¿por el IMC? ¿por los antecedentes familiares?

Los resultados son consistentes con la evidencia: glucosa plasmática en ayunas es el predictor dominante (alineado con los criterios ADA), seguido de IMC y edad. Nada sorprendente — pero eso es precisamente la validación que un CDSS necesita para que un clínico confíe en él.

---

## Lo que el modelo no puede hacer

Esto es igual de importante que lo que sí puede hacer:

- El dataset es mujeres adultas de herencia Pima Indian. No es generalizable a otras poblaciones sin validación externa.
- No hay datos de medicación, actividad física ni dieta. Los confundidores más importantes están ausentes.
- La glucosa en el dataset es de PTOG, no glucemia en ayunas estándar. El contexto de medición importa.
- Un AUC de 0.83 es bueno estadísticamente. No es suficiente para uso clínico sin validación prospectiva e implementación institucional.

Publicar las limitaciones no es modestia. Es lo que diferencia a alguien que entiende el problema clínico de alguien que solo sabe entrenar modelos.

---

## La app

Construí una interfaz en Streamlit que permite introducir los parámetros de un paciente y obtener:
- Probabilidad de riesgo con categorización (bajo / moderado / alto)
- Gráfico SHAP local mostrando qué variables contribuyeron en ese paciente concreto
- Valores de referencia clínicos para orientar la entrada de datos

No es un producto. Es una demostración de arquitectura CDSS: input clínico estructurado → modelo → explicabilidad → output accionable.

---

## Repositorio y demo

Todo el código está en GitHub, con notebooks documentados desde EDA hasta deployment. El README está escrito para que lo entienda tanto un data scientist como un médico sin experiencia en ML.

Si trabajas en HealthTech, Clinical AI, o Medical Affairs y te parece relevante, me interesa mucho tu feedback — especialmente sobre la parte de explicabilidad y su potencial aplicación real.

[GitHub](https://github.com/Aram9574/diabetes-risk-cdss) · [Demo](https://your-streamlit-url.streamlit.app)

---

*Alejandro Zakzuk — Médico | IA Aplicada a la Sanidad (CEMP) | Salud Digital (Universidad Europea de Madrid)*
