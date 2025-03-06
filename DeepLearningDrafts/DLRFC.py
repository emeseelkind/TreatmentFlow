"""
TreatmentFlow
Random Forest: Patient Priority Prediction

Automate assignment of patient priority (1-5) using Triage information 
Training based on 500k+ observation dataset of triage information associated with patient priority

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence

Random Forest supervised classification model for patient triage and diagnosis prediction
"""

"""
Step 1: Import Libraries
sklearn libraries used for data preprocessing and machine learning model building
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
# For building the Random Forest model
from sklearn.ensemble import RandomForestClassifier
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
    print(DataFrame.isnull().sum().sum(), "total missing values in the dataset")

    # Using ESI score as our target for patient priority (1-5)
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
    
    # Drop columns not needed for model training like non-medical demographic data
    exclude_cols = ['dep_name','esi','lang','religion','maritalstatus','employstatus','insurance_status']  # Add any other columns to exclude
    feature_cols = [col for col in DataFrame.columns if col not in exclude_cols]

    # Split numeric and categorical features
    num_features = DataFrame[feature_cols].select_dtypes(include=[np.number])
    cat_features = DataFrame[feature_cols].select_dtypes(include=['object', 'category', 'bool'])
    print(f"Using {len(num_features.columns)} numeric features and {len(cat_features.columns)} categorical features")

    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, num_features.columns),
                      ('cat', categorical_transformer, cat_features.columns)])
    
    x = DataFrame[feature_cols]  # Features
    y = DataFrame['esi'].values  # Target (Emergency Severity Index)
    
    # Handle NaN values in y
    if np.isnan(y).any():
        print("\nWarning: y contains NaN values. Imputing with most frequent value.")
        y_imputer = SimpleImputer(strategy='most_frequent')
        y = y_imputer.fit_transform(y.reshape(-1, 1)).ravel()

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Apply preprocessing
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    
    print(f"\nPreprocessed training data shape: {x_train.shape}")
    print(f"\nPreprocessed test data shape: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test , preprocessor

"""
Step 3: Build the Model. 
    select a suitable machine learning algorithm, 
"""
def build_model(x_train, x_test, y_train, y_test):
    """Build the Random Forest model for priority prediction"""
    
    start_time = time.time()

    # Create a RandomForestClassifier model
    model = RandomForestClassifier(
        n_estimators=100,           # Number of trees in the forest
        max_depth=None,             # No limit to depth of trees
        random_state=42,            # Random seed for reproducibility
        n_jobs=-1,                  # Use all CPU cores for training
        verbose=True                # Display progress during training
    )
    print("\nModel created:")
    print(model)
    return model , start_time

"""
Step 4: Train the Model. 
    train the model on the training data, 
"""
def train_model(model, start_time, x_train, y_train):
    """Train the Random Forest model"""
    
    # Train the model
    model.fit(x_train, y_train)
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    print("\nModel training complete.")
    # Save the trained model
    joblib.dump(model, 'patient_priority_model_rf.pkl')
    print("\nModel saved as patient_priority_model_rf.pkl")
    return model

"""
Step 5: Evaluate the Model. 
    - Evaluate its performance on a separate test set, 
"""
def evaluate_model(model, start_time, x_test, y_test):
    """Evaluate the Random Forest model"""
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
    plt.ylabel('Actual Priority (ESI)')
    plt.xlabel('Predicted Priority (ESI)')
    plt.savefig('confusion_matrix_rf.png')
    plt.close()
    print("\nConfusion matrix saved as 'confusion_matrix_rf.png'")

    # Save the model
    joblib.dump(model, 'patient_priority_model_rf.pkl')
    print("\nModel saved as patient_priority_model_rf.pkl")

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
    print("\nWelcome to the TreatmentFlow Random Forest module!")
    print("\n -------------------------------------------------------------------------")
    print("\nLoading data.")
    file_path = 'new_hospital_data_75.csv'
    DataFrame = load_data(file_path)
    print("\n -------------------------------------------------------------------------")
    print("\nPreprocessing data.")
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data(DataFrame)
    # 5 levels of patient priority
    priority_mapping = {
            1: "Immediate (Life Threatening)",
            2: "Emergency (Very Urgent)",
            3: "Urgent",
            4: "Semi-Urgent",
            5: "Non-Urgent"
        }
    # Build the model
    print("\n -------------------------------------------------------------------------")
    print("\nBuilding Random Forest model.")
    model , start_time = build_model(x_train, x_test, y_train, y_test)
    print("\n -------------------------------------------------------------------------")
    print("\nTraining model.")
    model = train_model(model, start_time, x_train, y_train)
    print("\n -------------------------------------------------------------------------")
    print("\nEvaluating model.")
    model = evaluate_model(model, start_time, x_test, y_test)
    print("\n -------------------------------------------------------------------------")
    p# Predict patient priority for sample patients
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
