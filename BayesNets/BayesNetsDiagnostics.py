import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(filepath, symptom_cols=None):
    """
    Load triage data and preprocess it for Bayesian network analysis.
    
    Parameters:
    filepath (str): Path to the CSV file containing triage data
    symptom_cols (list): List of symptom column names to consider. If None, all columns will be used.
    
    Returns:
    pd.DataFrame: Preprocessed dataframe
    """
    # Load the data
    triage_data = pd.read_csv(filepath)
    
    # If no specific symptom columns are provided, use all except ID and target columns
    if symptom_cols is None:
        # Assuming the last column is the diagnosis and first column might be ID
        symptom_cols = triage_data.columns[1:-1]
    
    # Extract only the needed columns
    df = triage_data[list(symptom_cols) + ['diagnosis']]
    
    # Handle missing values
    df = df.fillna(0)
    
    # Ensure all symptom values are binary (0 or 1)
    for col in symptom_cols:
        df[col] = df[col].astype(int).clip(0, 1)
    
    return df

def create_symptom_indices(df, diagnosis_col='diagnosis'):
    """
    Create dictionaries mapping symptoms to indices and diagnoses to indices.
    
    Parameters:
    df (pd.DataFrame): Preprocessed triage data
    diagnosis_col (str): Name of the diagnosis column
    
    Returns:
    tuple: (symptom_to_idx, idx_to_symptom, diagnosis_to_idx, idx_to_diagnosis)
    """
    symptom_cols = [col for col in df.columns if col != diagnosis_col]
    
    # Create mappings
    symptom_to_idx = {symptom: i for i, symptom in enumerate(symptom_cols)}
    idx_to_symptom = {i: symptom for symptom, i in symptom_to_idx.items()}
    
    # Create diagnosis mappings
    unique_diagnoses = df[diagnosis_col].unique()
    diagnosis_to_idx = {diagnosis: i for i, diagnosis in enumerate(unique_diagnoses)}
    idx_to_diagnosis = {i: diagnosis for diagnosis, i in diagnosis_to_idx.items()}
    
    return symptom_to_idx, idx_to_symptom, diagnosis_to_idx, idx_to_diagnosis

def split_data(df, test_size=0.2, random_state=42, diagnosis_col='diagnosis'):
    """
    Split data into training and testing sets.
    
    Parameters:
    df (pd.DataFrame): Preprocessed triage data
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility
    diagnosis_col (str): Name of the diagnosis column
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[diagnosis_col])
    y = df[diagnosis_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_bayesian_network(X_train, y_train, symptom_to_idx, diagnosis_to_idx):
    """
    Build a Bayesian Network using PyMC3 for disease diagnosis.
    
    Parameters:
    X_train (pd.DataFrame): Training features (symptoms)
    y_train (pd.Series): Training labels (diagnoses)
    symptom_to_idx (dict): Mapping from symptom names to indices
    diagnosis_to_idx (dict): Mapping from diagnosis names to indices
    
    Returns:
    tuple: (trace, model)
    """
    n_symptoms = len(symptom_to_idx)
    n_diagnoses = len(diagnosis_to_idx)
    
    # Convert diagnoses to numerical indices
    y_train_idx = y_train.map(diagnosis_to_idx)
    
    with pm.Model() as model:
        # Prior probabilities for each diagnosis
        diagnosis_prior = pm.Dirichlet('diagnosis_prior', 
                                       a=np.ones(n_diagnoses),
                                       shape=(n_diagnoses,))
        
        # Conditional probability of each symptom given each diagnosis
        symptom_given_diagnosis = pm.Beta('symptom_given_diagnosis',
                                          alpha=1, beta=1,
                                          shape=(n_diagnoses, n_symptoms))
        
        # For each patient in the training set
        for i, (_, symptoms) in enumerate(X_train.iterrows()):
            # The diagnosis for this patient
            diagnosis = pm.Categorical(f'diagnosis_{i}', 
                                       p=diagnosis_prior, 
                                       observed=y_train_idx.iloc[i])
            
            # For each symptom
            for symptom_name, has_symptom in symptoms.items():
                symptom_idx = symptom_to_idx[symptom_name]
                
                # Likelihood of observing this symptom given the diagnosis
                pm.Bernoulli(f'symptom_{i}_{symptom_name}',
                             p=symptom_given_diagnosis[diagnosis, symptom_idx],
                             observed=has_symptom)
        
        # Sample from the posterior distribution
        trace = pm.sample(1000, tune=1000, cores=1, return_inferencedata=True)
    
    return trace, model

def predict_diagnosis(X_test, trace, symptom_to_idx, idx_to_diagnosis):
    """
    Predict diagnoses for test data using the trained Bayesian Network.
    
    Parameters:
    X_test (pd.DataFrame): Test features (symptoms)
    trace (pm.MultiTrace): Samples from the posterior distribution
    symptom_to_idx (dict): Mapping from symptom names to indices
    idx_to_diagnosis (dict): Mapping from diagnosis indices to names
    
    Returns:
    tuple: (predicted_diagnoses, diagnosis_probabilities)
    """
    n_diagnoses = len(idx_to_diagnosis)
    n_samples = len(trace.posterior['diagnosis_prior'][0])
    
    # Get posterior samples of parameters
    diagnosis_prior_samples = trace.posterior['diagnosis_prior'].values.mean(axis=(0, 1))
    symptom_given_diagnosis_samples = trace.posterior['symptom_given_diagnosis'].values.mean(axis=(0, 1))
    
    predicted_diagnoses = []
    diagnosis_probabilities = []
    
    # For each patient in the test set
    for _, symptoms in X_test.iterrows():
        # Initialize probabilities
        probs = diagnosis_prior_samples.copy()
        
        # Update probabilities based on observed symptoms
        for symptom_name, has_symptom in symptoms.items():
            symptom_idx = symptom_to_idx[symptom_name]
            
            for d in range(n_diagnoses):
                # P(symptom | diagnosis) if the patient has the symptom
                p_symptom_given_d = symptom_given_diagnosis_samples[d, symptom_idx]
                
                if has_symptom:
                    probs[d] *= p_symptom_given_d
                else:
                    probs[d] *= (1 - p_symptom_given_d)
        
        # Normalize probabilities
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        
        # Store results
        predicted_diagnosis = idx_to_diagnosis[np.argmax(probs)]
        predicted_diagnoses.append(predicted_diagnosis)
        diagnosis_probabilities.append(probs)
    
    return predicted_diagnoses, diagnosis_probabilities

def generate_diagnosis_report(patient_symptoms, diagnosis_probs, idx_to_diagnosis, idx_to_symptom, threshold=0.1):
    """
    Generate a diagnosis probability report for a doctor.
    
    Parameters:
    patient_symptoms (pd.Series): Patient's symptoms
    diagnosis_probs (np.ndarray): Probability of each diagnosis
    idx_to_diagnosis (dict): Mapping from diagnosis indices to names
    idx_to_symptom (dict): Mapping from symptom indices to names
    threshold (float): Probability threshold to include in the report
    
    Returns:
    dict: Report containing diagnosis probabilities and supporting evidence
    """
    report = {
        "patient_symptoms": {symptom: (1 if value else 0) for symptom, value in patient_symptoms.items()},
        "diagnosis_probabilities": [],
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Sort diagnoses by probability
    sorted_indices = np.argsort(diagnosis_probs)[::-1]
    
    # Include diagnoses above threshold
    for idx in sorted_indices:
        prob = diagnosis_probs[idx]
        if prob >= threshold:
            diagnosis_name = idx_to_diagnosis[idx]
            report["diagnosis_probabilities"].append({
                "diagnosis": diagnosis_name,
                "probability": float(prob),
                "confidence": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            })
    
    return report

def evaluate_model(y_test, predicted_diagnoses):
    """
    Evaluate the performance of the Bayesian Network model.
    
    Parameters:
    y_test (pd.Series): True diagnoses
    predicted_diagnoses (list): Predicted diagnoses
    
    Returns:
    dict: Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_test, predicted_diagnoses)
    report = classification_report(y_test, predicted_diagnoses, output_dict=True)
    
    evaluation = {
        "accuracy": accuracy,
        "classification_report": report
    }
    
    return evaluation

def visualize_symptom_relationships(trace, idx_to_symptom, idx_to_diagnosis):
    """
    Visualize relationships between symptoms and diagnoses.
    
    Parameters:
    trace (pm.MultiTrace): Samples from the posterior distribution
    idx_to_symptom (dict): Mapping from symptom indices to names
    idx_to_diagnosis (dict): Mapping from diagnosis indices to names
    """
    # Get the mean symptom probabilities given each diagnosis
    symptom_given_diagnosis = trace.posterior['symptom_given_diagnosis'].values.mean(axis=(0, 1))
    
    plt.figure(figsize=(14, 10))
    
    # Plot heatmap of symptom probabilities given each diagnosis
    plt.imshow(symptom_given_diagnosis, aspect='auto', cmap='viridis')
    plt.colorbar(label='P(Symptom | Diagnosis)')
    
    # Label the axes
    plt.xlabel('Symptoms')
    plt.ylabel('Diagnoses')
    plt.xticks(range(len(idx_to_symptom)), list(idx_to_symptom.values()), rotation=90)
    plt.yticks(range(len(idx_to_diagnosis)), list(idx_to_diagnosis.values()))
    
    plt.title('Symptom-Diagnosis Relationship Heatmap')
    plt.tight_layout()
    plt.savefig('symptom_diagnosis_heatmap.png')
    plt.close()

def save_diagnosis_probabilities(df, diagnosis_probabilities, idx_to_diagnosis, output_file='diagnosis_probabilities.csv'):
    """
    Save diagnosis probabilities for all patients to a CSV file.
    
    Parameters:
    df (pd.DataFrame): Original dataframe with patient data
    diagnosis_probabilities (list): List of arrays with diagnosis probabilities
    idx_to_diagnosis (dict): Mapping from diagnosis indices to names
    output_file (str): Path to save the CSV file
    """
    # Create a dataframe with diagnosis probabilities
    probs_df = pd.DataFrame(index=df.index)
    
    # Add each diagnosis as a column
    for d_idx, d_name in idx_to_diagnosis.items():
        probs_df[f"{d_name}_probability"] = [probs[d_idx] for probs in diagnosis_probabilities]
    
    # Add the original data
    result_df = pd.concat([df, probs_df], axis=1)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Diagnosis probabilities saved to {output_file}")

def apply_treatment_guidelines(diagnosis, severity):
    """
    Apply treatment guidelines based on diagnosis and severity.
    
    Parameters:
    diagnosis (str): The diagnosed condition
    severity (str): Severity level (Mild, Moderate, Severe)
    
    Returns:
    dict: Treatment guidelines
    """
    # This is a simplified version - in practice, would connect to a medical guidelines database
    treatment_map = {
        "Pneumonia": {
            "Mild": ["Oral antibiotics", "Rest", "Hydration"],
            "Moderate": ["IV antibiotics", "Oxygen therapy", "Hospitalization"],
            "Severe": ["Broad-spectrum IV antibiotics", "ICU monitoring", "Ventilation if needed"]
        },
        "Influenza": {
            "Mild": ["Antiviral medication", "Rest", "Hydration"],
            "Moderate": ["Antiviral medication", "Symptom management", "Close monitoring"],
            "Severe": ["Hospitalization", "IV antivirals", "Respiratory support"]
        },
        # Add more conditions as needed
        "Default": {
            "Mild": ["Symptomatic treatment", "Follow-up in 3 days"],
            "Moderate": ["Close monitoring", "Targeted therapy", "Follow-up in 2 days"],
            "Severe": ["Immediate specialist consultation", "Hospitalization consideration"]
        }
    }
    
    # Get treatment guidelines
    if diagnosis in treatment_map:
        return {
            "diagnosis": diagnosis,
            "severity": severity,
            "recommended_treatments": treatment_map[diagnosis].get(severity, treatment_map["Default"][severity]),
            "followup_time": "24 hours" if severity == "Severe" else "72 hours" if severity == "Moderate" else "1 week"
        }
    else:
        return {
            "diagnosis": diagnosis,
            "severity": severity,
            "recommended_treatments": treatment_map["Default"][severity],
            "followup_time": "48 hours" if severity in ["Moderate", "Severe"] else "1 week"
        }

def determine_severity(patient_data, diagnosis):
    """
    Determine the severity of a condition based on patient data.
    
    Parameters:
    patient_data (dict): Patient's data including vital signs
    diagnosis (str): The diagnosed condition
    
    Returns:
    str: Severity level (Mild, Moderate, Severe)
    """
    # This is a simplified version - in practice, would use more complex criteria
    severity_points = 0
    
    # Check vital signs if available
    if "vitals" in patient_data:
        vitals = patient_data["vitals"]
        
        # High fever
        if vitals.get("temperature", 37) > 39:
            severity_points += 2
        elif vitals.get("temperature", 37) > 38:
            severity_points += 1
        
        # Abnormal heart rate
        if vitals.get("heart_rate", 70) > 120 or vitals.get("heart_rate", 70) < 50:
            severity_points += 2
        elif vitals.get("heart_rate", 70) > 100 or vitals.get("heart_rate", 70) < 60:
            severity_points += 1
        
        # Low oxygen saturation
        if vitals.get("oxygen_saturation", 98) < 90:
            severity_points += 3
        elif vitals.get("oxygen_saturation", 98) < 94:
            severity_points += 1
        
        # Abnormal blood pressure
        if vitals.get("systolic_bp", 120) > 180 or vitals.get("systolic_bp", 120) < 90:
            severity_points += 2
    
    # Check risk factors
    risk_factors = patient_data.get("risk_factors", [])
    severity_points += len(risk_factors)
    
    # Check symptom duration
    if patient_data.get("symptom_duration", 0) > 7:
        severity_points += 1
    
    # Determine severity level
    if severity_points >= 5:
        return "Severe"
    elif severity_points >= 2:
        return "Moderate"
    else:
        return "Mild"

def main(triage_data_path, symptom_cols=None):
    """
    Main function to run the entire pipeline.
    
    Parameters:
    triage_data_path (str): Path to the triage data CSV file
    symptom_cols (list): List of symptom column names to consider
    
    Returns:
    tuple: (evaluation, model, trace)
    """
    # Load and preprocess data
    df = load_and_preprocess_data(triage_data_path, symptom_cols)
    
    # Create indices
    symptom_to_idx, idx_to_symptom, diagnosis_to_idx, idx_to_diagnosis = create_symptom_indices(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Build Bayesian Network
    print("Building Bayesian Network...")
    trace, model = build_bayesian_network(X_train, y_train, symptom_to_idx, diagnosis_to_idx)
    
    # Predict diagnoses for test set
    print("Predicting diagnoses...")
    predicted_diagnoses, diagnosis_probabilities = predict_diagnosis(
        X_test, trace, symptom_to_idx, idx_to_diagnosis
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = evaluate_model(y_test, predicted_diagnoses)
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    
    # Visualize symptom relationships
    print("Visualizing symptom relationships...")
    visualize_symptom_relationships(trace, idx_to_symptom, idx_to_diagnosis)
    
    # Save diagnosis probabilities
    save_diagnosis_probabilities(X_test, diagnosis_probabilities, idx_to_diagnosis)
    
    # Generate a sample report for the first test patient
    sample_patient = X_test.iloc[0]
    sample_probs = diagnosis_probabilities[0]
    report = generate_diagnosis_report(sample_patient, sample_probs, idx_to_diagnosis, idx_to_symptom)
    
    # Determine severity and apply treatment guidelines
    sample_patient_data = {
        "vitals": {
            "temperature": 38.5,
            "heart_rate": 95,
            "oxygen_saturation": 96,
            "systolic_bp": 130
        },
        "risk_factors": ["diabetes"],
        "symptom_duration": 3
    }
    severity = determine_severity(sample_patient_data, predicted_diagnoses[0])
    treatment = apply_treatment_guidelines(predicted_diagnoses[0], severity)
    
    print("\nSample Diagnosis Report:")
    print(f"Diagnosis: {predicted_diagnoses[0]}")
    print(f"Severity: {severity}")
    print(f"Recommended treatments: {', '.join(treatment['recommended_treatments'])}")
    print(f"Follow-up time: {treatment['followup_time']}")
    
    return evaluation, model, trace

if __name__ == "__main__":
    # Example usage
    print("TreatmentFlow: Bayesian Networks for Disease Diagnostics")
    print("-------------------------------------------------------")
    
    # Define symptom columns (these would be actual columns from your data)
    example_symptoms = [
        "fever", "cough", "shortness_of_breath", "fatigue", 
        "headache", "sore_throat", "body_aches", "runny_nose",
        "nausea", "vomiting", "diarrhea", "loss_of_taste",
        "chest_pain", "abdominal_pain"
    ]
    
    # Run the main function with the triage data
    # Replace 'triage_data.csv' with actual data path
    print("Note: Replace 'triage_data.csv' with your actual triage data file path.")
    print("Loading sample data...")
    
    try:
        evaluation, model, trace = main("triage_data.csv", example_symptoms)
        print("\nProcessing complete!")
    except FileNotFoundError:
        print("\nError: Triage data file not found.")
        print("Please provide a valid file path to your triage data CSV file.")
        print("The CSV should contain columns for symptoms and a 'diagnosis' column.")