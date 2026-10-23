import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Mahim\Downloads\archive\Heart.csv.csv")
print("=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

# Display data type and unique values for each column
for column in df.columns:
    print(f"\nColumn: {column}")
    print(f"Data Type: {df[column].dtype}")
    print(f"Unique Values: {df[column].unique()}")
    print(f"Number of Unique Values: {df[column].nunique()}")