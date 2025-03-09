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

symptoms = diagnostics[0][:-1]  # list of symptoms
diagnostics = diagnostics[1:]   # list of diagnostic observations (with prognosis)

print(symptoms)

