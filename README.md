# ❤️ Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Decision%20Tree%20Classifier-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An interactive, full-stack Machine Learning decision support web application built with **Flask**, **Decision Tree Machine Learning Classifier**, **Bootstrap 5**, and **Chart.js**.

Heart Disease Prediction System evaluates 13 key cardiac health biomarkers to deliver real-time risk predictions, probabilistic confidence metrics, interactive analytics dashboards, and downloadable PDF clinical reports.

---

## 🌟 Key Features & Portfolio Highlights

- **🤖 Decision Tree Machine Learning Model**: Decision Tree Classifier trained on 1,000+ patient records achieving **97.56% test accuracy**.
- **🚩 Highlighted Top Risk Banner**: Result page features prominent top-level highlighted risk status banner.
- **🎨 Interactive Home Page**: Features a live quick-risk preview calculator widget, animated heartbeat telemetry, and step-by-step diagnostic workflow.
- **⚡ 1-Click Clinical Test Presets**: Includes quick preset profiles (*Low Risk*, *High Risk*, *Moderate*) allowing instant test predictions without manual form input.
- **📊 Interactive Analytics Dashboard**: Powered by Chart.js for real-time visualization of risk class distribution, age demographic breakdown, and chest pain classification.
- **📄 PDF Clinical Report Generator**: Built-in ReportLab document engine allowing users to download custom clinical reports.
- **📜 Prediction Audit Log**: Historical logging of patient assessments with client-side text searching, risk filtering, and log clearing actions.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Machine Learning** | Scikit-Learn, Decision Tree Classifier, Joblib, Pandas, NumPy, Matplotlib |
| **Backend** | Python 3.14, Flask, ReportLab (PDF Generation), CSV persistence |
| **Frontend** | HTML5, Vanilla CSS3, JavaScript (ES6+), Bootstrap 5 |
| **Visualizations** | Chart.js, FontAwesome 6, Google Fonts (*Outfit*, *Inter*) |

---

## 📂 Project Directory Structure

```text
heart-disease-prediction/
├── app.py                      # Main Flask Web Server & API Routes
├── save_model.py               # Decision Tree Training & Joblib Serialization Script
├── train_model.py              # Model Evaluation Script
├── model_comparison.py         # Benchmark Comparison Graphic Script
├── feature_importance.py       # Feature Importance Graphic Script
├── confusion_matrix_plot.py    # Confusion Matrix Graphic Script
├── dataset/
│   └── heart.csv               # Clinical Dataset (1027 records, 14 columns)
├── model/
│   └── heart_model.pkl         # Trained Decision Tree Model Binary
├── static/
│   ├── css/
│   │   └── style.css           # Custom Design System
│   └── images/                 # Model Evaluation Graphics
└── templates/                  # Jinja2 Dynamic HTML Templates
    ├── base.html               # Master Layout Template
    ├── index.html              # Interactive Home Page with Live Quick Risk Widget
    ├── predict.html            # Assessment Form with 1-Click Presets
    ├── result.html             # Highlighted Risk Banner Top & Recommendations Bottom
    ├── dashboard.html          # Interactive Chart.js Dashboard
    ├── model.html              # Decision Tree Model Specs & Graphics
    ├── history.html            # Audit History & Search Log
    ├── about.html              # Portfolio & Methodology Page
    ├── 404.html                # Custom 404 Error Page
    └── 500.html                # Custom 500 Error Page
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+ installed on your system.

### 2. Installation & Setup
Navigate to the repository folder:

```bash
cd heart-disease-prediction
```

Install required Python packages:

```bash
pip install flask pandas numpy scikit-learn joblib matplotlib reportlab
```

### 3. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to: **`http://localhost:5000`**

---

## 🩺 Evaluated Clinical Biomarkers

1. **Age**: Patient age in years.
2. **Sex**: Biological sex (1 = Male, 0 = Female).
3. **Chest Pain (CP)**: Angina classification (0: Typical, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic).
4. **Resting BP (trestbps)**: Systolic blood pressure in mmHg at rest.
5. **Cholesterol (chol)**: Serum cholesterol in mg/dL.
6. **Fasting Blood Sugar (fbs)**: Fasting blood sugar > 120 mg/dL (1 = True, 0 = False).
7. **Resting ECG (restecg)**: Resting electrocardiographic results.
8. **Max Heart Rate (thalach)**: Maximum heart rate achieved during exercise.
9. **Exercise Angina (exang)**: Exercise-induced angina (1 = Yes, 0 = No).
10. **Oldpeak**: ST depression induced by exercise relative to rest.
11. **Slope**: Slope of peak exercise ST segment.
12. **CA**: Number of major vessels (0–3) colored by fluoroscopy.
13. **Thal**: Thalassemia status (1: Normal, 2: Fixed defect, 3: Reversible defect).

---

## 🛡️ License & Medical Disclaimer

This project is open-source under the MIT License.

> **Disclaimer**: This project is engineered strictly for educational, scientific research, and developer portfolio demonstration purposes.
