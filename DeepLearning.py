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
import numpy as np
import matplotlib.pyplot as plt
# seaborn: statistical data visualization
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

""" 
Step 1: Load and Preprocess the Data
"""
def load_and_preprocess_data(file_path):
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn information:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Drop rows with missing values or impute them
    # For this example, we'll drop rows with missing values
    df_clean = df.dropna()
    print(f"\nClean dataset shape after removing missing values: {df_clean.shape}")
    
    # Identify numerical and categorical features
    # This would depend on your actual data structure, adjust accordingly
    numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variables from feature lists
    if 'priority' in numerical_features:
        numerical_features.remove('priority')
    
    # Optional: Check the distribution of the target variable
    print("\nDistribution of priority levels:")
    print(df_clean['priority'].value_counts(normalize=True) * 100)
    
    # Let's visualize the priority distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='priority', data=df_clean)
    plt.title('Distribution of Patient Priority Levels')
    plt.ylabel('Count')
    plt.xlabel('Priority Level (1-5)')
    plt.savefig('priority_distribution.png')
    
    return df_clean, numerical_features, categorical_features

"""
file_path = 'C:/Users/emese/Desktop/TreatmentFlow/hospital_triage_patient_data.csv'
df = pd.read_csv(file_path)

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


"""
Main Execution Block
"""
def main():
    # File path
    file_path = 'new_hospital_data_75.csv'
    
    # Step 1: Load and preprocess data
    df, numerical_features, categorical_features = load_and_preprocess_data(file_path)
    print("DataFrame:\n", df.head())  # Print the first few rows of the DataFrame
    print("Numerical Features:", numerical_features)  # Print numerical features
    print("Categorical Features:", categorical_features)  # Print categorical features
