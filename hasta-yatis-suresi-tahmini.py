# ---------- Import Libraries ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------- Import Data and Perform EDA ----------

df = pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")
df_ = df.head(50)

df.info()

describe = df.describe()

los = df["Length of Stay"] 
df["Length of Stay"] = df ["Length of Stay"].replace("120 +", 120) 
df["Length of Stay"] = pd.to_numeric(df["Length of Stay"])
los = df[ "Length of Stay"]

df.isna().sum()

for column in df.columns:
    unique_values = len(df[column].unique())
    print(f"Number of unique values in {column}: {unique_values}")

df = df[df["Patient Disposition"] != "Expired"]

"""
hasta yatis suresi - age - type of admission - payment type
"""

f, ax = plt.subplots()
sns.boxplot(x="Payment Typology 1", y="Length of Stay", data=df)
plt.title("Payment Typology 1 vs Length of Stay")
plt.xticks(rotation=60)

f, ax = plt.subplots()
sns.countplot(x="Age Group", data=df[df["Payment Typology 1"] == "Medicare"], order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"])
plt.title("Medicare Patients for Age Group")

f, ax = plt.subplots()
sns.boxplot(x="Type of Admission", y="Length of Stay", data=df)
plt.title("Type of Admission vs Length of Stay")
plt.xticks(rotation=60)

f, ax = plt.subplots()
sns.boxplot(x="Age Group", y="Length of Stay", data=df, order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"])
plt.title("Age Group vs Length of Stay")
plt.xticks(rotation=60)

# ---------- Feature Encoding ----------

df = df.drop(["Hospital Service Area", "Hospital County", "Operating Certificate Number",
    "Facility Name", "Zip Code - 3 digits", "Patient Disposition", "Discharge Year",
    "CCSR Diagnosis Description", "CCSR Procedure Description", "APR DRG Description",
    "APR MDC Description", "APR Severity of Illness Description",
    "Payment Typology 2", "Payment Typology 3", "Birth Weight", "Total Charges", "Total Costs"], axis=1)

age_group_index = {"0 to 17": 1, "18 to 29": 2, "30 to 49": 3, "50 to 69": 4, "70 or Older": 5}
gender_index = {"U": 0, "F": 1, "M": 2}
risk_and_severity_index = {np.nan: 0, "Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}

df["Age Group"] = df["Age Group"].apply(lambda x: age_group_index[x])
df["Gender"] = df["Gender"].apply(lambda x: gender_index[x])
df["APR Risk of Mortality"] = df["APR Risk of Mortality"].apply(lambda x: risk_and_severity_index[x])

encoder = OrdinalEncoder()
df["Race"] = encoder.fit_transform(np.asarray(df["Race"]).reshape(-1, 1))
df["Ethnicity"] = encoder.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1, 1))
df["Type of Admission"] = encoder.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1, 1))
df["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1, 1))
df["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1, 1))
df["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1, 1))
df["Payment Typology 1"] = encoder.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1, 1))
df["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1, 1))

df.isna().sum()

















