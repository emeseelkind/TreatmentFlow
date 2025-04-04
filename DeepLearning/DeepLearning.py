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
import os


"""
Step 2: Load and Preprocess the Data. 
    collect labeled data relevant to your problem, 
    prepare the data by cleaning and preprocessing it
"""
def load_data(directory):
    combined_df = pd.DataFrame()
    
    # Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            if 'esi' in df.columns:
                print(f"Loaded {len(df)} records from {filename}")
                combined_df = pd.concat([combined_df, df])
            else:
                print(f"Warning: 'esi' column not found in {filename}, skipping file.")
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print("\nCTAS (Priority) Distribution in combined data:")
    print(combined_df['esi'].value_counts().sort_index())
    combined_df = combined_df.reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='esi', data=combined_df)
    plt.title('Distribution of CTAS (Canadian Triage and Acuity Scale)')
    plt.xlabel('CTAS  Score (1 = highest priority, 5 = lowest priority)')
    plt.ylabel('Number of Patients')
    plt.savefig('DeepLearning/CTAS_distribution.png')
    plt.close()
    return combined_df

def preprocess_data(DataFrame):
    """Preprocess data for model training"""
    # First, let's handle NaN values in the target variable (esi)
    print(f"Before cleaning, DataFrame shape: {DataFrame.shape}")
    print(f"Number of NaN values in 'esi' column: {DataFrame['esi'].isna().sum()}")
    
    # Remove rows with NaN values in the target variable
    DataFrame = DataFrame.dropna(subset=['esi'])
    print(f"After removing NaN values in 'esi', DataFrame shape: {DataFrame.shape}")
    
    # Convert 'esi' to int to ensure it's a proper class label
    DataFrame['esi'] = DataFrame['esi'].astype(int)
    
    # Continue with the rest of preprocessing
    exclude_cols = ['dep_name', 'esi', 'lang', 'religion', 'maritalstatus', 'employstatus', 'insurance_status']
    feature_cols = [col for col in DataFrame.columns if col not in exclude_cols]
    num_features = DataFrame[feature_cols].select_dtypes(include=[np.number])
    cat_features = DataFrame[feature_cols].select_dtypes(include=['object', 'category', 'bool'])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features.columns),
            ('cat', categorical_transformer, cat_features.columns)
        ]
    )
    
    x = DataFrame[feature_cols]
    y = DataFrame['esi'].values
    
    # Check for any remaining NaN values
    if np.isnan(y).any():
        raise ValueError("Target variable 'esi' still contains NaN values after preprocessing")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test, y_train, y_test, preprocessor
    
"""
Step 3: Build the Model. 
    select a suitable machine learning algorithm, 
"""
def build_model(x_train, x_test, y_train, y_test):
    """Build the deep learning model for priority prediction"""
    
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
    print("\nModel created:")
    print(model)
    return model , start_time
"""
Step 4: Train the Model. 
    train the model on the training data, 
"""
def train_model(model, start_time, x_train, y_train):
    """Train the deep learning model"""
    
    # Train the model
    model.fit(x_train, y_train)
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    print("\nModel training complete.")
    # Save the trained model
    joblib.dump(model, 'patient_priority_model.pkl')
    print("\nModel saved as patient_priority_model.pkl")
    return model

"""
Step 5: Evaluate the Model. 
    - Evaluate its performance on a separate test set, 
"""
def evaluate_model(model, start_time, x_test, y_test):
    """Evaluate the deep learning model"""
    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Create and save confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.title('Confusion Matrix for Patient Priority Prediction')
    plt.ylabel('Actual Priority (CTAS)')
    plt.xlabel('Predicted Priority (CTAS)')
    plt.savefig('DeepLearning/confusion_matrix.png')
    plt.close()
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

    # Save the model
    joblib.dump(model, 'DeepLearning/patient_priority_model.pkl')
    print("\nModel saved as patient_priority_model.pkl")

    # Print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal time taken: {total_time:.2f} seconds")
    return model
"""
Step 6: Use the Model. 
    - Deploy the model to make predictions on new data
"""
def predict_priority(model, sample_patients, preprocessor, priority_mapping):
    """Predict patient priority using the trained model"""
    print("\nPredicting patient priority for sample patients...")
    new_data_processed = preprocessor.transform(sample_patients)
    # Get predictions
    predictions = model.predict(new_data_processed)
    probabilities = model.predict_proba(new_data_processed)
    # Create a results dataframe
    results = pd.DataFrame({
        'predicted_priority': predictions,
        'priority_description': [priority_mapping[p] for p in predictions]
    })
    # Add probability for each priority level
    for i in range(1, 6):
        class_idx = model.classes_.tolist().index(i) if i in model.classes_ else -1
        if class_idx >= 0:
            results[f'probability_priority_{i}'] = probabilities[:, class_idx]
        else:
            results[f'probability_priority_{i}'] = 0.0
    
    return results

def main():
    # Initialize the model
    print("\nWelcome to the TreatmentFlow Deep Learning module!")
    print("\n -------------------------------------------------------------------------")
    print("\nLoading data...")
    
    directory = 'C:/Users/emese/Desktop/TreatmentFlow/CTAS_files'
    DataFrame = load_data(directory)
    print("\n -------------------------------------------------------------------------")
    print("\nPreprocessing data...")
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data(DataFrame)
    print("Data loading and preprocessing complete.")



    # 5 levels of patient priority
    priority_mapping = {
            1: "Immediate (Resuscitation)",
            2: "Emergency",
            3: "Urgent",
            4: "Less Urgent",
            5: "Non-Urgent"
        }
    # Build the model
    print("\n -------------------------------------------------------------------------")
    print("\nBuilding deep learning model...")
    model, start_time = build_model(x_train, x_test, y_train, y_test)
    # Train the model
    print("\n -------------------------------------------------------------------------")
    print("\nTraining deep learning model...")
    model = train_model(model, start_time, x_train, y_train)

    # Evaluate the model    
    print("\n -------------------------------------------------------------------------")
    print("\nEvaluating deep learning model...")
    model = evaluate_model(model, start_time, x_test, y_test)

    # Predict patient priority for sample patients
    print("\n -------------------------------------------------------------------------")
    print("\nPredicting patient priority for sample patients.")
    sample_indices = np.random.choice(range(len(DataFrame)), size=5, replace=False)
    sample_patients = DataFrame.drop('esi', axis=1).iloc[sample_indices]
    results = predict_priority(model, sample_patients, preprocessor, priority_mapping)

    # Display results
    print("\nSample Patient Predictions:")
    print(results[['predicted_priority', 'priority_description']])
    
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