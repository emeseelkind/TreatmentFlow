import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load the dataset
# -----------------------------
def load_esi_data(esi_directory='C:/Users/emese/Desktop/TreatmentFlow/ESI_Files'):
    """
    Load data from separate ESI files and combine them into a single DataFrame
    """
    print(f"Loading ESI data from {esi_directory}...")
    
    combined_df = pd.DataFrame()
    
    # Loop through ESI levels 1-5
    for esi_level in range(1, 6):
        file_path = f"{esi_directory}/esi{esi_level}.csv"
        
        if os.path.exists(file_path):
            print(f"Loading ESI level {esi_level} data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Check if the ESI values match the expected level
            if 'esi' in df.columns:
                # Verify that all rows have the correct ESI value
                if not all(df['esi'] == esi_level):
                    print(f"Warning: Not all rows in {file_path} have ESI = {esi_level}")
                    print(f"ESI value counts in file: {df['esi'].value_counts()}")
                    # Ensure all rows have the correct ESI value
                    df = df[df['esi'] == esi_level]
                
                print(f"Loaded {len(df)} records with ESI level {esi_level}")
                combined_df = pd.concat([combined_df, df])
            else:
                print(f"Error: 'esi' column not found in {file_path}")
        else:
            print(f"Error: File {file_path} not found")
    
    # Verify the final combined DataFrame
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print("\nESI (Priority) Distribution in combined data:")
    print(combined_df['esi'].value_counts().sort_index())
    
    # Create visualization of ESI distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='esi', data=combined_df)
    plt.title('Distribution of ESI (Emergency Severity Index)')
    plt.xlabel('ESI Score (1 = highest priority, 5 = lowest priority)')
    plt.ylabel('Number of Patients')
    plt.savefig('esi_distribution.png')
    plt.close()
    print("\nCheck esi_distribution.png for the graph.")
    
    return combined_df


def preprocess_data(DataFrame):
    """Preprocess data for model training"""
    
    # Drop columns not needed for model training like non-medical demographic data
    exclude_cols = ['dep_name','esi','lang','religion','maritalstatus','employstatus','insurance_status']  # Add any other columns to exclude
    feature_cols = [col for col in DataFrame.columns if col not in exclude_cols]

    # Split numeric and categorical features (use numpy)
    num_features = DataFrame[feature_cols].select_dtypes(include=[np.number])
    cat_features = DataFrame[feature_cols].select_dtypes(include=['object', 'category', 'bool'])
    print(f"Using {len(num_features.columns)} numeric features and {len(cat_features.columns)} categorical features")

    # Create preprocessing pipelines for numeric and categorical features
    # Pipeline allows you to sequentially apply a list of transformers to preprocess the data
    print("\nApplying preprocessing pipelines...")
    
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [0, 1])  # Use column indices (e.g., 0 and 1)
        ]
    )
    
    x = DataFrame[feature_cols]  # Features
    y = DataFrame['esi'].values  # Target (Emergency Severity Index)
    
    # Handle NaN values in y
    if np.isnan(y).any():
        print("\nWarning: y contains NaN values. Imputing with most frequent value.")
        y_imputer = SimpleImputer(strategy='most_frequent')
        y = y_imputer.fit_transform(y.reshape(-1, 1)).ravel()

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print("\nApplying transformations")
    # Apply preprocessing - preprocessor from sklearn
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    
    print(f"\nPreprocessed training data shape: {x_train.shape}")
    print(f"\nPreprocessed test data shape: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test, preprocessor


# -----------------------------
# 4. Build Classification Model
# -----------------------------

def build_model(preprocessor):
    """Build the model for priority prediction"""
    
    # Create a RandomForestClassifier
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight="balanced"))
    ])
    return model


def train_model(model, x_train, y_train):
    """Train the model"""
    model.fit(x_train, y_train)
    return model


# -----------------------------
# 5. Predictions & Evaluation
# -----------------------------
def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(x_test)

    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # -----------------------------
    # 6. Save the Model
    # -----------------------------
    joblib.dump(model, "esi_classifier.pkl")
    print("Model saved as esi_classifier.pkl")

def predict_priority(model, sample_patients, preprocessor, priority_mapping):
    """Predict priority levels for sample patients"""
    predictions = model.predict(sample_patients)
    prob_predictions = model.predict_proba(sample_patients)
    
    # Create a DataFrame to hold predictions
    results = pd.DataFrame(predictions, columns=['predicted_priority'])
    
    # Map predictions to priority descriptions
    results['priority_description'] = results['predicted_priority'].map(priority_mapping)
    
    # Add probability columns
    for i, priority in enumerate(priority_mapping.keys(), 1):
        results[f'probability_priority_{priority}'] = prob_predictions[:, i - 1]
    
    return results


def main():
    # Initialize the model
    print("\nWelcome to the TreatmentFlow Deep Learning module!")
    print("\n -------------------------------------------------------------------------")
    print("\nLoading data from separate ESI files.")
    DataFrame = load_esi_data()
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
    print("\nBuilding model.")
    model = build_model(preprocessor)

    # Train the model
    print("\n -------------------------------------------------------------------------")
    print("\nTraining model.")
    model = train_model(model, x_train, y_train)

    # Evaluate the model
    print("\n -------------------------------------------------------------------------")
    print("\nEvaluating model.")
    evaluate_model(model, x_test, y_test)

    # Predict patient priority for sample patients
    print("\n -------------------------------------------------------------------------")
    print("\nPredicting patient priority for sample patients.")
    sample_patients = pd.DataFrame()
    for esi_level in range(1, 6):
        level_data = DataFrame[DataFrame['esi'] == esi_level]
        if not level_data.empty:
            sample = level_data.drop('esi', axis=1).sample(1)
            sample_patients = pd.concat([sample_patients, sample])

    if len(sample_patients) < 5:
        additional_samples = 5 - len(sample_patients)
        more_samples = DataFrame.drop('esi', axis=1).sample(additional_samples)
        sample_patients = pd.concat([sample_patients, more_samples])

    results = predict_priority(model, sample_patients, preprocessor, priority_mapping)

    print("\nSample Patient Predictions:")
    print(results[['predicted_priority', 'priority_description']])
    
    print("\nProbability breakdown for each priority level:")
    probability_cols = [col for col in results.columns if 'probability_priority' in col]
    print(results[probability_cols])
    
    print("\nTreatmentFlow Deep Learning module completed successfully!")


if __name__ == "__main__":
    main()
