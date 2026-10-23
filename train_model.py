# Step 1: Import Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")

# Step 2: Load Dataset
df = pd.read_csv(DATASET_PATH)

print("="*60)
print("HEART DISEASE PREDICTION")
print("="*60)

# Step 3: Display Basic Information
print("\nDataset Shape:", df.shape)

# Step 4: Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Step 5: Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Step 6: Separate Features and Target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")