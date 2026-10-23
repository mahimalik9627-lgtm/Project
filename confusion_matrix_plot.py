import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "static", "images", "confusion_matrix.png")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load(MODEL_PATH)

# Plot confusion matrix
plt.style.use('default')
fig, ax = plt.subplots(figsize=(6, 5), facecolor='#ffffff')
ax.set_facecolor('#ffffff')

disp = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=["Low Risk", "High Risk"],
    cmap="Blues",
    ax=ax
)

plt.title("Decision Tree Confusion Matrix", color='#0f172a', fontsize=14, pad=15, fontweight='bold')
plt.tick_params(colors='#0f172a', labelsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Confusion Matrix Saved to {OUTPUT_PATH}!")