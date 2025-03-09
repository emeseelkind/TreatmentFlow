"""
TreatmentFlow
Bayesian Networks - Disease Diagnostics from Triage Inputs

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

# this file is an implementation of Bayes' Nets used to generate disease probability documents for doctors

import csv

# importing symptom data file
diagnostics = []
with open("BayesNets/symbipredict_2022.csv") as symptom_info:
    reader = csv.reader(symptom_info)
    for row in reader:
        diagnostics.append(row)

diagnoses = [row[-1] for row in diagnostics[1:]]        # list of possible diagnoses
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
