"""
TreatmentFlow
Deep Learning: Patient Priority Prediction

Automate assignment of patient priority (1-5) using Triage information 
Training based on 500k+ observation dataset of triage information associated with patient priority

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence

DNN (Deep Neural Network) supervised classification model for patient triage and diagnosis prediction
"""

"""
Step 1: Import Libraries. 
Step 2: Load and Preprocess the Data. 
    collect labeled data relevant to your problem, 
    prepare the data by cleaning and preprocessing it, 

Step 3: Build the Model. 
    select a suitable machine learning algorithm, 

Step 4: Train the Model. 
    train the model on the training data, 

Step 5: Evaluate the Model. 
    evaluate its performance on a separate test set, 
    and finally deploy the model to make predictions on new data
"""
"""
Step 1: Import Libraries
sklearn and tensorflow libraries used for data preprocessing and deep learning model building
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Splitting Data into Training & Test Sets
from sklearn.model_selection import train_test_split
# Feature Scaling & Encoding Categorical Variables
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    
    print(f"Loading data from {file_path}...")
    # Pandas reading DataFrame from CSV file 
    DataFrame = pd.read_csv(file_path)
    # Pandas .shape = (rows, columns)
    print(f"Dataset shape: {DataFrame.shape}")
    # Display dataset information
    print("\nData Overview:")
    print(DataFrame.head(6))
    print("\nData Types:")
    print(DataFrame.dtypes.value_counts())
    print("\nMissing Values:")
    # print(DataFrame.isnull().sum(), "missing values per column") 
    print(DataFrame.isnull().sum().sum(), "total missing values in the dataset")

    # Using ESI score as our target for patient priority (1-5)
    # ESI = Emergency Severity Index, where 1 is most severe and 5 is least severe
    print("\nESI (Priority) Distribution:")
    print(DataFrame['esi'].value_counts().sort_index())
    plt.figure(figsize=(10, 6))
    sns.countplot(x='esi', data=DataFrame)
    plt.title('Distribution of ESI (Emergency Severity Index)')
    plt.xlabel('ESI Score (1 = highest priority, 5 = lowest priority)')
    plt.ylabel('Number of Patients')
    plt.savefig('esi_distribution.png')
    plt.close()

def main():
    # Initialize the model
    file_path = 'new_hospital_data_75.csv'
    load_data(file_path)
    priority_mapping = {
            1: "Immediate (Life Threatening)",
            2: "Emergency (Very Urgent)",
            3: "Urgent",
            4: "Semi-Urgent",
            5: "Non-Urgent"
        }
if __name__ == "__main__":
    main()