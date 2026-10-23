import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load Dataset
df = pd.read_csv("dataset/heart.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Create Model
model = DecisionTreeClassifier(random_state=42)

# 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("=" * 50)
print("Cross Validation Scores")
print(scores)

print("=" * 50)
print("Average Accuracy :", scores.mean() * 100)

print("=" * 50)
print("Maximum Accuracy :", scores.max() * 100)

print("=" * 50)
print("Minimum Accuracy :", scores.min() * 100)