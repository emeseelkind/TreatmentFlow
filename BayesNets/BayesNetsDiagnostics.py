"""
TreatmentFlow
Bayesian Networks - Disease Diagnostics from Triage Inputs

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

# this file is an implementation of Bayes' Nets used to generate disease probability documents for doctors

import csv
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from collections import defaultdict

def data_preprocessing(file_path):
    diagnostics = []
    with open(file_path) as symptom_info:
        reader = csv.reader(symptom_info)
        for row in reader:
            diagnostics.append(row)

    diagnoses = [row[-1] for row in diagnostics[1:]] # list of possible diagnoses
    diagnosis_options = []  # list of each possible diagnosis (without duplicates)

    i = 0
    while i < len(diagnoses):
        if diagnoses[i] not in diagnosis_options:
            diagnosis_options.append(diagnoses[i])
        i += 1

    symptoms = diagnostics[0][:-1] # list of all symptom variables
    symptom_bools = [row[:-1] for row in diagnostics[1:]]   # 2d array of booleans referring to symptoms

    # goal with CPTs: get the conditional probabilities of each diagnosis given symptoms

    # print(len(diagnoses))
    print(diagnosis_options)
    print(len(diagnosis_options))
    # print(len(symptoms))
    # print(len(symptom_bools[0]))
    
    # Convert the data to a pandas DataFrame for easier handling
    data = []
    for i, symptom_row in enumerate(symptom_bools):
        row_data = {}
        for j, symptom in enumerate(symptoms):
            # Convert string boolean to integer (0 or 1)
            row_data[symptom] = 1 if symptom_row[j].lower() in ['true', '1', 't', 'yes', 'y'] else 0
        row_data['diagnosis'] = diagnoses[i]
        data.append(row_data)
    
    df = pd.DataFrame(data)
    
    # Clean column names (remove spaces, special chars)
    df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').lower() for col in df.columns]
    symptoms = [symptom.strip().replace(' ', '_').replace('(', '').replace(')', '').lower() for symptom in symptoms]
    
    return df, symptoms, diagnosis_options

# Step 2: Define the Bayesian Network structure
def create_bayesian_network(df):
    """
    Create a naive Bayesian network using the diagnostics file information
    """
    # Define edges: diagnosis -> each symptom
    edges = [('diagnosis', symptom) for symptom in symptoms]
    
    # Create Bayesian Network with defined edges
    model = BayesianNetwork(edges)
    
    print(f"Created Bayesian Network with {len(edges)} edges")
    return model


# Step 3: Train the Bayesian Network
def train_model(model, df):
    """
    Train the Bayesian Network using Maximum Likelihood or Bayesian estimation.
    MaximumLikelihoodEstimator from pgmpy.estimators import MaximumLikelihoodEstimator
    """
    # Use Maximum Likelihood Estimator to estimate the CPDs
    estimator = MaximumLikelihoodEstimator(model, df)
    
    # Initialize CPDs for each node
    for node in model.nodes():
        print(f"Estimating CPD for {node}")
        cpd = estimator.estimate_cpd(node)
        model.add_cpds(cpd)
    
    # Check if the model is valid
    if model.check_model():
        print("Model is valid and trained successfully")
    else:
        print("Error: Model is not valid")
    
    return model


# Step 4: Create an inference engine
def create_inference_engine(model):
    """
    Create an inference engine for the Bayesian Network.
    posterior probabilistic inference
    VariableElimination
    """
    inference_engine = VariableElimination(model)
    return inference_engine


# Step 5: Generate diagnosis probabilities
def generate_diagnosis_probabilities(inference_engine, symptoms, evidence):
    """
    Generate probabilities for possible diagnoses given observed symptoms.
    
    Parameters:
    inference_engine: The inference engine
    symptoms: List of all symptom variables
    evidence: Dictionary of observed symptoms {symptom_name: value (0 or 1)}
    
    Returns:
    DataFrame with diagnosis probabilities
    """
    pass


# Step 6: Generate a comprehensive report for doctors
def generate_doctor_report(probabilities, evidence, df):
    """
    Generate a comprehensive report for doctors with diagnosis probabilities
    and relevant symptom information.
    
    Parameters:
    probabilities: DataFrame with diagnosis probabilities
    evidence: Dictionary of observed symptoms
    df: Original dataframe with all data
    
    Returns:
    Report as a string
    """
    pass


# Main function to run the entire pipeline
def run_diagnostic_system(file_path, observed_symptoms):
    """
    Run the entire diagnostic system pipeline.
    
    Parameters:
    file_path: Path to the dataset file
    observed_symptoms: List of observed symptoms as strings
    
    Returns:
    Diagnosis report
    """
    pass


# for each possible diagnosis, we want:
    # P(Di | S1, S2, ..., Sn)


# goal for Bayes Net: make every independent pair of symptoms siblings, rather than parents/children
    # X and Y are independent if
        # for all x, y: P(x,y) = P(x) * P(y) = P(x) * P(y|x)
        # for all x, y: P(x|y) = P(x)
    # conditional independence of X and Y, given Z
        # for all x, y, z: P(x, y|z) = P(x|z) * P(y|z)
        # for all x, y, z: P(x|z, y) = P(x|z)     

    # each node in the BN refers to a probability table
        # if node A has parents X and Y, A refers to the probability table P(A | X,Y)
            # P(xi | x1, ..., x_(i-1)) = P(xi | parents(Xi)

    # bayes net should allow us to output the posterior probability of a diagnosis D given some observed evidence S
        # P(D|S) = (P(S|D) * P(D)) / P(S)
        # display the ~10 most probable diagnoses given a set of evidence (boolean symptom values)


def main():
    print("\nData Preprocessing:\n")
    file_path = "BayesNets/symbipredict_2022.csv"
    diagnoses, diagnosis_options, symptoms, symptom_bools = data_preprocessing(file_path)
    print("\nBayesian Network Creation:\n")
    model = create_bayesian_network(symptoms, symptom_bools)
    print("\nTraining the Bayesian Network:\n")
    train_model(model, symptom_bools)
    print("\nCreating the Inference Engine:\n")
    inference_engine = create_inference_engine(model)
    print("\nGenerating Diagnosis Probabilities:\n")

    #observed_symptoms = testing set from list of symptoms./// or new patient input
    probabilities = generate_diagnosis_probabilities(inference_engine, symptoms, observed_symptoms)
    print("\nGenerating Doctor Report:\n")

    report = generate_doctor_report(probabilities, observed_symptoms)
    print(report)
    run_diagnostic_system(file_path, observed_symptoms)
    

if __name__ == "__main__":
    main()