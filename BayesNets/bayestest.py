"""import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df.apply(lambda x: x.astype('category').cat.codes)
    return df


def learn_structure(data):
    hc = HillClimbSearch(data)
    model_structure = hc.estimate(scoring_method=BicScore(data))
    return model_structure


def build_and_train_model(structure, data):
    model = BayesianNetwork(structure.edges())
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
    return model


def predict_diagnosis(model, evidence, target_variable="Diagnosis"):
    infer = VariableElimination(model)
    prediction = infer.map_query(variables=[target_variable], evidence=evidence)
    return prediction


def main():
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    file_path = "../TreatmentFlow/BayesNets/symbipredict_2022.csv"
    data = load_and_preprocess_data(file_path)
    print("Data loaded and preprocessed.")
    
    structure = learn_structure(data)
    print("Model structure learned.")
    model = build_and_train_model(structure, data)
    print("Model built and trained.")

    #randomly take a line from the csv file and use those symptoms to predict the diagnosis
    evidence = data.sample(1).iloc[0].to_dict()
    # Remove the target variable from evidence
    evidence.pop("Diagnosis", None)
    print("Evidence for prediction:", evidence)

    prediction = predict_diagnosis(model, evidence)
    print("Prediction made.")
    print("Diagnosis Prediction:", prediction)


if __name__ == "__main__":
    main()
"""

import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination

# Load the data
data = pd.read_csv('symbipredict_2022.csv')

# Optional: Feature selection to reduce dimensionality (too many symptoms can make structure learning slow)
# Here we could use mutual information or chi-square to select top symptoms

# Structure learning - discover relationships between symptoms and diseases
hc = HillClimbSearch(data)
model = hc.estimate(scoring_method=BicScore(data))

# Convert the learned structure to a Bayesian Network
bn_model = BayesianNetwork(model.edges())

# Fit parameters (CPTs) to the data
bn_model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')

# Create an inference object
inference = VariableElimination(bn_model)

# Example: Diagnostic inference (given symptoms, what's the probability of each disease?)
def diagnose(symptoms):
    """
    symptoms: dict of symptom names and values (0 or 1)
    returns: probability distribution over diseases
    """
    return inference.query(variables=['prognosis'], evidence=symptoms)

# Example usage:
diagnosis = diagnose({
    'itching': 1, 
    'skin_rash': 1, 
    'nodal_skin_eruptions': 1,
    'dischromic_patches': 1
})
print(diagnosis)