"""
TreatmentFlow
Deep Learning: Patient Priority Prediction

Automate assignment of patient priority (1-5) using Triage information 
Training based on 500k+ observation dataset of triage information associated with patient priority

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
​"""


"""
Step 1: Import Libraries. 
Step 2: Load and Preprocess the Data. 
Step 3: Build the Model. 
Step 4: Train the Model. 
Step 5: Evaluate the Model. 
"""
import pandas as pd

# Replace 'path/to/file.csv' with the actual file path
file_path = 'C:/Users/emese/Desktop/TreatmentFlow/hospital_triage_patient_data.csv'
df = pd.read_csv(file_path)
"""
# Display the first 5 rows
print(df.head())

# Display the column names
print(df.columns)

# Get information about the DataFrame (e.g., number of rows, columns, and data types)
print(df.info())

# Get basic statistics about numerical columns
print(df.describe())
"""


# Extract the target variable (patient priority) and the features (triage information)
"""
Extract a specific column:
age_data = df['age']
print(age_data.head())

Extract multiple columns:
demographics = df[['age', 'gender', 'ethnicity']]
print(demographics.head())

Extract a specific row by index:
first_row = df.iloc[0]
print(first_row)

Filter rows based on a condition:
older_patients = df[df['age'] > 50]
print(older_patients.head())
"""
