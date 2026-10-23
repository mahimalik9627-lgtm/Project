import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\Mahim\Downloads\archive\Heart.csv.csv")

#Target Distribution
target_count = df["target"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(["No Disease", "Heart Disease"], target_count)

plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")

plt.show()

#Age Distribution
plt.figure(figsize=(8,5))

plt.hist(df["age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()

#gender Distribution
gender = df["sex"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(["Female", "Male"], gender)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()


#Cholesterol Distribution
plt.figure(figsize=(8,5))

plt.hist(df["chol"], bins=15)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Patients")

plt.show()

#Heart Disease by Gender
male = df[df["sex"] == 1]["target"].value_counts()
female = df[df["sex"] == 0]["target"].value_counts()

plt.figure(figsize=(7,5))

plt.bar(["Male-No", "Male-Yes"], male)

plt.title("Heart Disease in Males")
plt.ylabel("Patients")

plt.show()

plt.figure(figsize=(7,5))

plt.bar(["Female-No", "Female-Yes"], female)

plt.title("Heart Disease in Females")
plt.ylabel("Patients")

plt.show()

#Correlation Matrix
correlation = df.corr()

plt.figure(figsize=(10,8))

plt.imshow(correlation)

plt.colorbar()

plt.xticks(range(len(correlation.columns)),
           correlation.columns,
           rotation=90)

plt.yticks(range(len(correlation.columns)),
           correlation.columns)

plt.title("Correlation Matrix")

plt.show()