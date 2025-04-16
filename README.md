# TreatmentFlow

**TreatmentFlow** is an AI-based system designed to optimize the flow of patients through the emergency room. It integrates three major AI components:
- **Deep Learning-based Triage:** Automatically assigns patient priority using a trained neural network.
- **Constraint Optimization for Bed Assignment:** Assigns hospital beds based on incoming patients and resource availability.
- **Bayesian Network Diagnostics:** Generates bedside documents for doctors, displaying the probabilities of specific conditions based on observed symptoms.

TreatmentFlow was originally developed by **Adam Neto** and **Emese Elkind** from February to April 2025 as a project for **CISC 352 - Artificial Intelligence**, a third-year course at Queen’s University School of Computing.

**Usage instructions:**
- Install all required dependencies (see below).
- Option 1: Run the TreatmentFlow_Lite.py file for access to our streamlined text-based UI for interacting with all 3 components of TreatmentFlow.
  - **Deep Learning Component (uses DeepLearning/DLRaw.py)**
    - Used to predict the a patient's CTAS priority level, whether or not they have performed a manual triage
  - **Constraint Satisfaction Component (uses CSP/HospitalClasses.py and CSP/GreedyBedAssignment.py)**
    - Used to assign patients to beds based on arrival time and priority
    - Also used to print the bed assignment through minute-by-minute simulated updates
  - **Bayesian Network Component (uses BayesNets/BNRaw.py)**
    - Used to generate patient diagnostic information based on the triage inputs (manually added or randomized) of the survey system
    - Displays diagnostic information in the console
    - Generates a patient diagnostic document in the bedside_documents folder

- Option 2: Run each component independently
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
