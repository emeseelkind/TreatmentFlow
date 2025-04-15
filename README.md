# TreatmentFlow

**TreatmentFlow** is an AI-based system designed to optimize the flow of patients through the emergency room. It integrates three major AI components:
- **Deep Learning-based Triage:** Automatically assigns patient priority using a trained neural network.
- **Constraint Optimization for Bed Assignment:** Assigns hospital beds based on incoming patients and resource availability.
- **Bayesian Network Diagnostics:** Generates bedside documents for doctors, displaying the probabilities of specific conditions based on observed symptoms.

TreatmentFlow was originally developed by **Adam Neto** and **Emese Elkind** from February to April 2025 as a project for **CISC 352 - Artificial Intelligence**, a third-year course at Queen’s University School of Computing.

**Usage instructions:**
- Install all required dependencies (see below).
- Run the TreatmentFlow_Lite.py file for access to our streamlined text-based UI for interacting with all 3 components of TreatmentFlow. 
  - **Deep Learning Component (DeepLearningMLP.py)**
    - Generates the confusion matrix and displays model accuracy for patient priority.
  - **Constraint Satisfaction Component (MIPBedAssignment.py)**
    - Compares solutions from both Mixed-Integer Programming (MIP) and a Greedy algorithm
    - To test Greedy-only scheduling:
      1. Create a hospital using the HospitalRecords class with at least 1 bed.
      2. Generate patients using the gen_patient_list() method.
      3. Initialize a scheduler with the Schedule class using the hospital object.
      4. Run run_hospital() and optionally waiting_times() to view Greedy algorithm results.
  - **Bayesian Network Component (BayesSKlearn.py)**
    - Generates the confusion matrix and displays model accuracy for diagnosis prediction.
    - Takes symptom inputs and outputs likely diagnoses and related probabilities.

# Installation Instructions

Install the following Python packages before running the project:
```
python -m pip install ortools
pip install pandas pyreadr seaborn scikit-learn pymc3 matplotlib
```
