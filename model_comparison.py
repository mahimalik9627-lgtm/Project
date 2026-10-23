import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "static", "images", "model_comparison.png")

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

# Models
models = {
    "Decision Tree\n(Current Model)": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM Classifier": SVC(probability=True),
    "KNN Classifier": KNeighborsClassifier()
}

results = {}

# Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    results[name] = accuracy
    print(f"{name} : {accuracy * 100:.2f}%")

# Clean Modern Light Plot
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='#ffffff')
ax.set_facecolor('#ffffff')

# Highlight Decision Tree with vibrant primary medical blue
colors_list = ['#0284c7', '#64748b', '#94a3b8', '#94a3b8', '#94a3b8']
bars = ax.bar(results.keys(), [v * 100 for v in results.values()], color=colors_list, width=0.48)

# Remove top, right, and left spines for a ultra-clean look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#cbd5e1')
ax.spines['bottom'].set_linewidth(1.2)

plt.ylabel("Accuracy (%)", color='#475569', fontsize=11, fontweight='bold')
plt.ylim(50, 108)
plt.xticks(color='#0f172a', fontsize=10.5, fontweight='bold')
plt.yticks(color='#64748b', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3, color='#cbd5e1')

for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2.0,
        yval + 2.0,
        f"{yval:.1f}%",
        ha='center',
        va='bottom',
        color='#0f172a',
        fontweight='bold',
        fontsize=11
    )

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=250, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()

print(f"Model Comparison Plot Saved to {OUTPUT_PATH}!")