import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\Mahim\Downloads\archive\Heart.csv.csv")

print("=" * 60)
print("HEART DISEASE DATASET ANALYSIS")
print("=" * 60)

# Display first 5 rows
print("\n1. First 5 Rows")
print(df.head())

# Display last 5 rows
print("\n2. Last 5 Rows")
print(df.tail())

# Dataset dimensions
print("\n3. Dataset Shape")
print(df.shape)

# Column names
print("\n4. Column Names")
print(df.columns.tolist())

# Data information
print("\n5. Dataset Information")
print(df.info())

# Missing values
print("\n6. Missing Values")
print(df.isnull().sum())

# Duplicate rows
print("\n7. Duplicate Rows")
print(df.duplicated().sum())

# Statistical summary
print("\n8. Statistical Summary")
print(df.describe())

# Target distribution
print("\n9. Target Distribution")
print(df["target"].value_counts())