import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\Mahim\Downloads\archive\Heart.csv.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Display data types
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display summary statistics
print("\nStatistical Summary:")
print(df.describe())

# Display target value counts
print("\nTarget Distribution:")
print(df["target"].value_counts())