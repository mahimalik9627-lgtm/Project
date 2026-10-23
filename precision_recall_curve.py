import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "static", "images", "roc_curve.png")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load Dataset
df = pd.read_csv(DATASET_PATH)

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction Probability
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0b0f19')
ax.set_facecolor('#0b0f19')

plt.plot(fpr, tpr, color='#00e699', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--')

plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', color='#94a3b8', fontsize=11)
plt.ylabel('True Positive Rate', color='#94a3b8', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC)', color='#ffffff', fontsize=14, pad=15, fontweight='bold')
plt.legend(loc="lower right", facecolor='#1e293b', edgecolor='#334155')
plt.grid(True, linestyle='--', alpha=0.2)
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"ROC Curve Saved to {OUTPUT_PATH}!")