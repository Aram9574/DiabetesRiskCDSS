# Diabetes Risk Prediction — Clinical ML Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Clinical decision support tool for type 2 diabetes risk stratification in primary care settings.**
> Built as a portfolio project demonstrating end-to-end clinical ML: from raw data to deployed application.

---

## Clinical Context

Type 2 diabetes affects over 537 million adults worldwide (IDF, 2021), with a large proportion undiagnosed at the time of presentation. Early risk stratification in primary care can enable targeted interventions — lifestyle modification, metabolic monitoring, and timely referral — before overt hyperglycemia develops.

This project builds a machine learning pipeline to support that decision: given a patient's basic clinical and anthropometric data, estimate their probability of diabetes and surface the variables driving that risk.

**This tool does not diagnose diabetes.** It is designed as a decision support layer for clinicians, consistent with the CDSS (Clinical Decision Support System) paradigm — augmenting clinical judgment, not replacing it.

---

## What This Project Demonstrates

| Layer | What it shows |
|---|---|
| Clinical framing | Ability to translate a clinical problem into an ML task with appropriate outcome definitions and metric selection |
| ML pipeline | End-to-end workflow: EDA, preprocessing, model comparison, hyperparameter tuning, evaluation |
| Explainability | SHAP-based local and global explanations with clinical interpretation |
| Deployment | Streamlit app with clinically-informed UX: reference ranges, risk categories, per-patient explanation |
| Communication | Readable documentation oriented to both clinical and technical audiences |

---

## Dataset

**Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

**Population:** Adult women of Pima Indian heritage, age ≥ 21 years.

**Target variable:** Binary — diabetes diagnosis confirmed by WHO criteria (plasma glucose ≥ 200 mg/dL or FPG ≥ 126 mg/dL on two occasions).

**Features used:**

| Feature | Clinical meaning |
|---|---|
| Pregnancies | Number of gestational events (proxy for gestational diabetes history) |
| Glucose | Plasma glucose at 2h in oral glucose tolerance test (mg/dL) |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skin fold thickness (mm) — proxy for subcutaneous fat |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Family history of diabetes — genetic risk score |
| Age | Age in years |

### Known Limitations (Clinically Important)

- **Single-sex, single-ethnicity sample.** Generalizability to other populations is uncertain. Do not apply thresholds derived here to general practice without validation.
- **Zero-value artifacts.** Biologically implausible zeros in Glucose, BMI, BloodPressure, etc. represent missing data, not true zeros. Handled explicitly in preprocessing (see notebook 02).
- **Cross-sectional snapshot.** No longitudinal follow-up; incident diabetes vs. prevalent diabetes distinction is not possible.
- **No medication or lifestyle data.** Confounders like antihyperglycemic treatment, physical activity, or dietary patterns are absent.

---

## Model Performance

| Model | AUC-ROC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| SVM (RBF) | — | — | — | — |
| **Random Forest** | **0.942** | — | — | **85.7%** |

> Full evaluation in `notebooks/03_evaluation.ipynb`, including calibration curves, confusion matrices, and threshold analysis.

**Why AUC-ROC as primary metric?**
In a screening context, we want a model that correctly identifies patients at risk (high sensitivity), even at some cost to specificity. AUC-ROC captures discrimination ability across all thresholds, making it the appropriate primary metric here. Accuracy alone would be misleading given class imbalance (~35% positive rate).

---

## Project Structure

```
diabetes-risk-cdss/
├── data/
│   └── diabetes.csv              # Raw dataset
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Feature engineering & imputation
│   └── 03_evaluation.ipynb       # Model comparison, SHAP, calibration
├── src/
│   ├── preprocessing.py          # Reusable preprocessing functions
│   ├── model.py                  # Model training & serialization
│   └── metrics.py                # Clinical metrics utilities
├── app/
│   └── streamlit_app.py          # Interactive risk calculator
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Aram9574/diabetes-risk-cdss.git
cd diabetes-risk-cdss
pip install -r requirements.txt

# Run notebooks
jupyter lab

# Run Streamlit app
streamlit run app/streamlit_app.py
```

---

## Live Demo

[Launch App](https://your-streamlit-url.streamlit.app) ← *link updated after deployment*

---

## Clinical Disclaimer

This tool is intended for educational and research purposes. It is not validated for clinical use and does not constitute medical advice. Any application in real clinical settings would require prospective validation, regulatory review, and institutional oversight.

---

## Author

**Alejandro Zakzuk** — Physician · AI Applied to Health (CEMP) · Digital Health (Universidad Europea de Madrid)

[LinkedIn](https://linkedin.com/in/alejandrozakzuk-ia-salud-digital) · [Website](https://alejandrozakzuk.com) · [GitHub](https://github.com/Aram9574)
