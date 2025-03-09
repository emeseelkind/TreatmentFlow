"""
TreatmentFlow
Bayesian Networks - Disease Diagnostics from Triage Inputs

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""
# Imports from scikit learn


"""
Input the symptoms (binary vector, 132 features)
Compute the posterior probability of each diagnosis given the symptoms:
𝑃(𝐷𝑖∣𝑆1,𝑆2,...,𝑆132)
Return the diagnoses with the highest probabilities
"""

"""
Step 2: Define Bayesian Network and CPTs
Step 3: Inference to compute posterior probabilities
"""




"""
Step 4: Bayesian optimization to tune model parameters:
Uses a Gaussian Process (GP) to model the probability distribution over possible parameter values.
Chooses the next best parameters to evaluate based on an acquisition function like Expected Improvement (EI).

"""


"""
Step 5: Generate a Bedside Report
After computing the top diagnoses with probabilities, generate a doctors-only summary.
The output can be:

Patient Condition Probability Table:
------------------------------------
1. Pneumonia - 85%
2. COVID-19 - 78%
3. Flu - 65%
4. ... 
"""