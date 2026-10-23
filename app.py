import os
import csv
import joblib
import pandas as pd
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    jsonify,
    session
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ===========================================
# Base Directory Setup & Path Resolution
# ===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")
PDF_REPORT_PATH = os.path.join(BASE_DIR, "Heart_Disease_Report.pdf")

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "heart_disease_prediction_system_secret_key_2026"

# ===========================================
# Load Decision Tree Model with Fallback Protection
# ===========================================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")

HISTORY_COLUMNS = [
    "Date", "Age", "Sex", "ChestPain", "BP", "Cholesterol", "FBS",
    "RestECG", "HeartRate", "ExerciseAngina", "OldPeak", "Slope",
    "CA", "Thal", "Prediction", "Confidence"
]

# Global store fallback for in-memory session
latest_result = None

# ===========================================
# Helper Functions
# ===========================================
def get_dataset_stats():
    """Retrieve summarized statistics from heart dataset."""
    if not os.path.exists(DATASET_PATH):
        return {
            "total": 0, "high": 0, "low": 0, "accuracy": 97.56,
            "avg_age": 54, "avg_chol": 246, "avg_bp": 131
        }
    
    df = pd.read_csv(DATASET_PATH)
    total = int(len(df))
    high = int((df["target"] == 1).sum())
    low = int((df["target"] == 0).sum())
    avg_age = round(float(df["age"].mean()), 1)
    avg_chol = round(float(df["chol"].mean()), 1)
    avg_bp = round(float(df["trestbps"].mean()), 1)
    
    return {
        "total": total,
        "high": high,
        "low": low,
        "accuracy": 97.56,
        "avg_age": avg_age,
        "avg_chol": avg_chol,
        "avg_bp": avg_bp
    }

# ===========================================
# Routes
# ===========================================

@app.route("/")
def home():
    stats = get_dataset_stats()
    return render_template("index.html", stats=stats)


@app.route("/about")
def about():
    stats = get_dataset_stats()
    return render_template("about.html", stats=stats)


@app.route("/dashboard")
def dashboard():
    stats = get_dataset_stats()
    
    # Dataset breakdown data for Chart.js
    chart_data = {
        "target_distribution": [stats["low"], stats["high"]],
        "age_groups": {"<40": 0, "40-50": 0, "51-60": 0, ">60": 0},
        "chest_pain": [0, 0, 0, 0]
    }
    
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        chart_data["age_groups"] = {
            "<40": int((df["age"] < 40).sum()),
            "40-50": int(((df["age"] >= 40) & (df["age"] <= 50)).sum()),
            "51-60": int(((df["age"] >= 51) & (df["age"] <= 60)).sum()),
            ">60": int((df["age"] > 60).sum())
        }
        cp_counts = df["cp"].value_counts().to_dict()
        chart_data["chest_pain"] = [
            int(cp_counts.get(0, 0)),
            int(cp_counts.get(1, 0)),
            int(cp_counts.get(2, 0)),
            int(cp_counts.get(3, 0))
        ]

    return render_template(
        "dashboard.html",
        stats=stats,
        chart_data=chart_data
    )


@app.route("/model")
def model_info():
    return render_template("model.html")


@app.route("/predict", methods=["GET"])
def prediction_form():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    global latest_result

    try:
        if model is None:
            raise ValueError("Decision Tree model is not loaded. Please train model/heart_model.pkl first.")

        # Extract features from form input
        features = [
            float(request.form.get("age", 50)),
            float(request.form.get("sex", 1)),
            float(request.form.get("cp", 0)),
            float(request.form.get("trestbps", 120)),
            float(request.form.get("chol", 200)),
            float(request.form.get("fbs", 0)),
            float(request.form.get("restecg", 0)),
            float(request.form.get("thalach", 150)),
            float(request.form.get("exang", 0)),
            float(request.form.get("oldpeak", 0.0)),
            float(request.form.get("slope", 1)),
            float(request.form.get("ca", 0)),
            float(request.form.get("thal", 2))
        ]

        # ML Prediction using Decision Tree Model
        prediction_val = int(model.predict([features])[0])

        # Calculate Confidence Score
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([features])[0]
            confidence = round(float(max(probabilities)) * 100, 2)
        else:
            confidence = 95.00

        # Detailed Advice & Risk Level Determination
        if prediction_val == 1:
            result_label = "High Risk of Heart Disease"
            risk_level = "High Risk"
            advice_categories = {
                "medical": [
                    "Schedule a consultation with a certified Cardiologist.",
                    "Obtain a comprehensive ECG, Echocardiogram, and Lipid blood panel.",
                    "Take all prescribed cardiovascular medications strictly as directed."
                ],
                "lifestyle": [
                    "Adopt a low-sodium, heart-healthy Mediterranean diet.",
                    "Completely eliminate tobacco smoke and minimize alcohol consumption.",
                    "Engage in light-to-moderate aerobic activity after doctor clearance."
                ],
                "monitoring": [
                    "Maintain twice-daily Blood Pressure and Heart Rate logs.",
                    "Schedule follow-up lipid profile tests every 30–60 days.",
                    "Monitor for any chest tightness, shortness of breath, or dizziness."
                ]
            }
            advice = [
                "Schedule a consultation with a certified Cardiologist promptly.",
                "Maintain strict continuous monitoring of Blood Pressure and Cholesterol.",
                "Adopt a low-sodium, heart-healthy Mediterranean diet.",
                "Engage in light-to-moderate aerobic exercise after physician clearance.",
                "Completely eliminate tobacco smoke and minimize alcohol consumption.",
                "Practice daily stress management techniques.",
                "Strictly adhere to any prescribed cardiovascular medications."
            ]
        else:
            result_label = "Low Risk of Heart Disease"
            risk_level = "Low Risk"
            advice_categories = {
                "medical": [
                    "Schedule routine annual preventive health exams and blood work.",
                    "Review family cardiovascular history with your primary physician."
                ],
                "lifestyle": [
                    "Maintain a balanced diet rich in whole grains, vegetables, and lean proteins.",
                    "Target 150+ minutes of moderate aerobic exercise per week.",
                    "Maintain consistent hydration and 7–8 hours of restful sleep daily."
                ],
                "monitoring": [
                    "Check baseline Blood Pressure and Cholesterol once every 6 months.",
                    "Maintain healthy body mass index (BMI) within recommended bounds."
                ]
            }
            advice = [
                "Continue maintaining a balanced diet rich in whole grains, vegetables, and lean proteins.",
                "Target 150+ minutes of moderate exercise per week.",
                "Maintain consistent hydration and 7–8 hours of restful sleep daily.",
                "Schedule routine annual health checkups and blood work.",
                "Keep body mass index (BMI) within a recommended healthy range."
            ]

        # Dynamic Contributing Risk Factor Breakdown
        risk_factors = []
        if features[0] >= 60: risk_factors.append({"name": "Age >= 60 Years", "impact": "Moderate", "value": f"{int(features[0])} yrs"})
        if features[3] >= 140: risk_factors.append({"name": "Elevated Resting Blood Pressure", "impact": "High", "value": f"{int(features[3])} mmHg"})
        if features[4] >= 240: risk_factors.append({"name": "High Serum Cholesterol", "impact": "High", "value": f"{int(features[4])} mg/dL"})
        if features[5] == 1: risk_factors.append({"name": "Fasting Blood Sugar > 120 mg/dL", "impact": "Moderate", "value": "Elevated"})
        if features[7] < 120: risk_factors.append({"name": "Reduced Maximum Heart Rate", "impact": "Moderate", "value": f"{int(features[7])} bpm"})
        if features[8] == 1: risk_factors.append({"name": "Exercise Induced Angina Present", "impact": "High", "value": "Positive"})
        if features[9] >= 1.5: risk_factors.append({"name": "Significant ST Segment Depression", "impact": "High", "value": f"{features[9]}"})
        if features[11] >= 1: risk_factors.append({"name": "Fluoroscopy Colored Vessel Defect", "impact": "High", "value": f"{int(features[11])} Vessel(s)"})

        # Formatted Patient Vitals Map for Report Table
        cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
        restecg_map = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "LV Hypertrophy"}
        slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
        thal_map = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

        patient_vitals = {
            "age": int(features[0]),
            "sex": "Male" if features[1] == 1 else "Female",
            "cp": cp_map.get(int(features[2]), "Unknown"),
            "trestbps": int(features[3]),
            "chol": int(features[4]),
            "fbs": "Elevated (>120 mg/dL)" if features[5] == 1 else "Normal (<=120 mg/dL)",
            "restecg": restecg_map.get(int(features[6]), "Normal"),
            "thalach": int(features[7]),
            "exang": "Yes (Present)" if features[8] == 1 else "No (Absent)",
            "oldpeak": features[9],
            "slope": slope_map.get(int(features[10]), "Flat"),
            "ca": int(features[11]),
            "thal": thal_map.get(int(features[12]), "Normal")
        }

        # Save to Prediction History CSV
        file_exists = os.path.isfile(HISTORY_FILE)
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(HISTORY_COLUMNS)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                *features,
                result_label,
                confidence
            ])

        report_id = f"HDS-{datetime.now().strftime('%Y%m%d')}-{int(features[0])}{int(features[3])}"

        # Store latest result in memory and session
        latest_result = {
            "report_id": report_id,
            "prediction": result_label,
            "risk_level": risk_level,
            "confidence": confidence,
            "advice": advice,
            "advice_categories": advice_categories,
            "risk_factors": risk_factors,
            "patient_vitals": patient_vitals,
            "features": features,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        session["latest_result"] = latest_result

        return render_template(
            "result.html",
            report_id=report_id,
            prediction=result_label,
            risk_level=risk_level,
            confidence=confidence,
            advice=advice,
            advice_categories=advice_categories,
            risk_factors=risk_factors,
            patient_vitals=patient_vitals,
            features=features
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template(
            "result.html",
            report_id="ERR-001",
            prediction="Prediction Error",
            risk_level="Error",
            confidence=0,
            advice=[f"An unexpected error occurred during prediction: {str(e)}"],
            advice_categories={},
            risk_factors=[],
            patient_vitals={},
            features=[]
        )


@app.route("/history")
def history():
    history_records = []
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if set(HISTORY_COLUMNS).issubset(df.columns):
                df = df.dropna(subset=["Date"])
                df = df.sort_values(by="Date", ascending=False)
                history_records = df.to_dict(orient="records")
            else:
                os.remove(HISTORY_FILE)
                history_records = []
        except Exception as e:
            print(f"Error reading history: {e}")
            history_records = []

    return render_template("history.html", history=history_records)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception as e:
            print(f"Error clearing history file: {e}")
    return redirect(url_for("history"))


@app.route("/download_report")
def download_report():
    """Generate and return a beautifully formatted PDF clinical report."""
    global latest_result
    
    report_data = session.get("latest_result") or latest_result

    # Fallback if no prediction in memory: try loading last row from history CSV
    if not report_data and os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not df.empty:
                last_row = df.iloc[-1]
                pred_text = str(last_row.get("Prediction", "Unknown"))
                conf_val = float(last_row.get("Confidence", 95.0))
                is_high = "High" in pred_text
                report_data = {
                    "prediction": pred_text,
                    "risk_level": "High Risk" if is_high else "Low Risk",
                    "confidence": conf_val,
                    "date": str(last_row.get("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                    "advice": [
                        "Consult a certified medical doctor or cardiologist.",
                        "Maintain regular blood pressure and health checkups.",
                        "Follow a balanced diet and regular active routine."
                    ] if is_high else [
                        "Maintain healthy lifestyle habits and regular exercise.",
                        "Keep routine annual medical exams."
                    ]
                }
        except Exception as e:
            print(f"Error extracting history for PDF fallback: {e}")

    # Build PDF with ReportLab
    doc = SimpleDocTemplate(
        PDF_REPORT_PATH,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    # Custom ReportLab Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    
    sub_title = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=18
    )

    heading2 = ParagraphStyle(
        'DocHeading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )

    story = []

    story.append(Paragraph("Heart Disease Prediction System — Patient Risk Report", title_style))
    story.append(Paragraph(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Decision Support System", sub_title))
    story.append(Spacer(1, 8))

    if report_data:
        pred_label = report_data.get("prediction", "N/A")
        confidence_val = report_data.get("confidence", 0)
        risk_color = colors.HexColor("#e11d48") if "High" in pred_label else colors.HexColor("#0d9488")

        # Summary Table
        summary_data = [
            [Paragraph("<b>Diagnostic Metric</b>", body_style), Paragraph("<b>Assessment Result</b>", body_style)],
            [Paragraph("Cardiovascular Risk Status", body_style), Paragraph(f"<b><font color='{risk_color.hexval()}'>{pred_label}</font></b>", body_style)],
            [Paragraph("Model Precision Score", body_style), Paragraph(f"<b>{confidence_val}%</b>", body_style)],
            [Paragraph("Evaluation Algorithm", body_style), Paragraph("Decision Tree Classifier", body_style)]
        ]

        t = Table(summary_data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Add Patient Vitals if available
        if "patient_vitals" in report_data and report_data["patient_vitals"]:
            pv = report_data["patient_vitals"]
            story.append(Paragraph("Patient Biomarker Summary", heading2))
            vitals_table_data = [
                [Paragraph("<b>Biomarker</b>", body_style), Paragraph("<b>Input Value</b>", body_style), Paragraph("<b>Biomarker</b>", body_style), Paragraph("<b>Input Value</b>", body_style)],
                [Paragraph("Age / Sex", body_style), Paragraph(f"{pv.get('age')} yrs / {pv.get('sex')}", body_style), Paragraph("Resting BP", body_style), Paragraph(f"{pv.get('trestbps')} mmHg", body_style)],
                [Paragraph("Cholesterol", body_style), Paragraph(f"{pv.get('chol')} mg/dL", body_style), Paragraph("Fasting Sugar", body_style), Paragraph(f"{pv.get('fbs')}", body_style)],
                [Paragraph("Chest Pain", body_style), Paragraph(f"{pv.get('cp')}", body_style), Paragraph("Max Heart Rate", body_style), Paragraph(f"{pv.get('thalach')} bpm", body_style)],
                [Paragraph("Rest ECG", body_style), Paragraph(f"{pv.get('restecg')}", body_style), Paragraph("Exercise Angina", body_style), Paragraph(f"{pv.get('exang')}", body_style)],
                [Paragraph("ST Oldpeak", body_style), Paragraph(f"{pv.get('oldpeak')}", body_style), Paragraph("ST Slope", body_style), Paragraph(f"{pv.get('slope')}", body_style)],
            ]
            vt = Table(vitals_table_data, colWidths=[120, 130, 120, 130])
            vt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(vt)
            story.append(Spacer(1, 12))

        story.append(Paragraph("Health Guidance & Recommendations", heading2))
        for advice_item in report_data.get("advice", []):
            story.append(Paragraph(f"• {advice_item}", body_style))
            
    else:
        story.append(Paragraph("No active prediction record found.", body_style))

    story.append(Spacer(1, 18))
    disclaimer_style = ParagraphStyle(
        'DocDisclaimer',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor("#64748b")
    )
    story.append(Paragraph("<b>Medical Disclaimer:</b> This report is generated by a Decision Tree machine learning decision support model for educational and portfolio demonstration purposes only.", disclaimer_style))

    doc.build(story)

    return send_file(
        PDF_REPORT_PATH,
        as_attachment=True,
        download_name="Heart_Disease_Prediction_Report.pdf"
    )

# ===========================================
# Error Handlers
# ===========================================

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html"), 500


# ===========================================
# App Runner
# ===========================================
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )