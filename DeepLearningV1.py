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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# For building the deep learning model
from sklearn.neural_network import MLPClassifier
# For evaluating the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
# For saving the model
import joblib
# For timing the training process
import time

"""
Step 2: Load and Preprocess the Data. 
    collect labeled data relevant to your problem, 
    prepare the data by cleaning and preprocessing it
"""
def load_data(file_path):
    
    print(f"Loading data from {file_path}...")
    # Pandas reading DataFrame from CSV file 
    DataFrame = pd.read_csv(file_path)
    # Pandas .shape = (rows, columns)
    print(f"Dataset shape: {DataFrame.shape}")
    # Display dataset information
    print("\nData Overview:")
    print(DataFrame.head(3))
    print("\nData Types:")
    print(DataFrame.dtypes.value_counts())
    print("\nMissing Values:")
    # print(DataFrame.isnull().sum(), "missing values per column") 
    print(DataFrame.isnull().sum().sum(), "total missing values in the dataset")

    # Using ESI score as our target for patient priority (1-5)
    # ESI = Emergency Severity Index, where 1 is most severe and 5 is least severe
    print("\nESI (Priority) Distribution:")
    print(DataFrame['esi'].value_counts().sort_index())
    print("\nCheck esi_distribution.png for the graph.")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='esi', data=DataFrame)
    plt.title('Distribution of ESI (Emergency Severity Index)')
    plt.xlabel('ESI Score (1 = highest priority, 5 = lowest priority)')
    plt.ylabel('Number of Patients')
    plt.savefig('esi_distribution.png')
    plt.close()
    return DataFrame

def preprocess_data(DataFrame):
    """Preprocess data for model training"""
    print("\nPreprocessing data...")
    # Drop columns not needed for model training like non-medical demographic data
    exclude_cols = ['dep_name','esi','lang','religion','maritalstatus','employstatus','insurance_status']  # Add any other columns to exclude
    feature_cols = []
    for col in DataFrame.columns:
        if col not in exclude_cols:
            feature_cols.append(col)
    # print(f"Feature columns: {feature_cols}") - works

    # Split numeric and categorical features (use numpy)
    num_features = DataFrame[feature_cols].select_dtypes(include=[np.number])
    cat_features = DataFrame[feature_cols].select_dtypes(include=['object', 'category', 'bool'])
    # print(f"Numeric features: {num_features.columns}") - works
    # print(f"Categorical features: {cat_features.columns}") - works
    print(f"Using {len(num_features.columns)} numeric features and {len(cat_features.columns)} categorical features")

    # Create preprocessing pipelines for numeric and categorical features
    # Pipeline allows you to sequentially apply a list of transformers to preprocess the data
    print("\nApplying preprocessing pipelines...")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    print("\nColumnTransformer")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features.columns),
            ('cat', categorical_transformer, cat_features.columns)])
    
    x = DataFrame[feature_cols]  # Features
    y = DataFrame['esi'].values  # Target (Emergency Severity Index)
    
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("\nfit_transform")
    # Apply preprocessing - preprocessor from sklearn
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    
    print(f"\nPreprocessed training data shape: {x_train.shape}")
    print(f"\nPreprocessed test data shape: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test , preprocessor
"""
Step 3: Build the Model. 
    select a suitable machine learning algorithm, 
"""
def build_model(x_train, y_train, x_test, y_test):
    """Build the deep learning model for priority prediction"""
    print("\nBuilding deep learning model...")
    # Create an MLPClassifier (Multi-Layer Perceptron - Neural Network)
    start_time = time.time()

    # MLPClassifier mimics a deep neural network architecture
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # Three hidden layers similar to the Keras model
        activation='relu',             # ReLU activation function
        solver='adam',                 # Adam optimizer
        alpha=0.0001,                  # L2 regularization parameter
        batch_size=32,                 # Mini-batch size
        learning_rate_init=0.001,      # Initial learning rate
        max_iter=100,                  # Maximum number of iterations
        early_stopping=True,           # Use early stopping
        validation_fraction=0.1,       # Use 10% of training data for validation
        n_iter_no_change=10,           # Number of iterations with no improvement to wait before early stopping
        random_state=42,               # Random seed for reproducibility
        verbose=True                   # Display progress during training
    )
    return model , start_time
"""
Step 4: Train the Model. 
    train the model on the training data, 
"""


def main():
    # Initialize the model
    file_path = 'new_hospital_data_75.csv'
    DataFrame = load_data(file_path)
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data(DataFrame)
    # 5 levels of patient priority
    priority_mapping = {
            1: "Immediate (Life Threatening)",
            2: "Emergency (Very Urgent)",
            3: "Urgent",
            4: "Semi-Urgent",
            5: "Non-Urgent"
        }
    model, start_time = build_model(x_train, y_train, x_test, y_test)
    # Train the model
    model, start_time = train_model(model, start_time, x_train, y_train)

    # Evaluate the model    
    model = evaluate_model(model, start_time, x_test, y_test)

    sample_indices = np.random.choice(range(len(DataFrame)), size=5, replace=False)
    sample_patients = DataFrame.drop('esi', axis=1).iloc[sample_indices]
    results = predict_priority(model, sample_patients, preprocessor, priority_mapping)

    # Display results
    print("\nSample Patient Predictions:")
    print(predictions[['predicted_priority', 'priority_description']])
    
    print("\nTreatmentFlow Deep Learning module completed successfully!")
    
    # Feature importance analysis (if available with the model)
    try:
        feature_importances = model.feature_importances_
        print("\nTop 10 Most Important Features:")
        # This would need additional code to map feature indices back to names
    except:
        print("\nFeature importance not available for this model type.")

    print("\nTreatmentFlow Deep Learning module completed successfully!")

if __name__ == "__main__":
    main()