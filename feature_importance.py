import os
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "static", "images", "feature_importance.png")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load trained model
model = joblib.load(MODEL_PATH)

# Feature names
features = [
    "Age",
    "Sex",
    "Chest Pain",
    "Resting BP",
    "Cholesterol",
    "Fasting BS",
    "Rest ECG",
    "Max HR",
    "Ex Angina",
    "Oldpeak",
    "ST Slope",
    "Major Vessels",
    "Thalassemia"
]

importance = model.feature_importances_

plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#ffffff')
ax.set_facecolor('#ffffff')

bars = ax.bar(features, importance, color='#0284c7', edgecolor='#0369a1', linewidth=1)

# Highlight top feature
max_idx = importance.argmax()
bars[max_idx].set_color('#e11d48')
bars[max_idx].set_edgecolor('#be123c')

plt.xticks(rotation=35, ha='right', color='#0f172a', fontsize=10)
plt.yticks(color='#0f172a', fontsize=10)
plt.title("Decision Tree Feature Importance", color='#0f172a', fontsize=14, pad=15, fontweight='bold')
plt.ylabel("Gini Importance Weight", color='#475569', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.3, color='#cbd5e1')
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Feature Importance Graph Saved Successfully to {OUTPUT_PATH}!")