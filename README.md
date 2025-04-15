# TreatmentFlow

TreatmentFlow is an AI-based project for optimizing the flow of patients through the emergency room.
It includes 3 major components:
- Deep Learning-based patient priority assignment through an automated triage system
- Constraint Optimization-based hospital bed assignment based on incoming patients and available resources
- Bayesian Network-based diagnostic tools to produce bedside documents for doctors only, which display the probabilities of certain conditions based on symptom inputs

TreatmentFlow was originally created by Adam Neto and Emese Elkind during the months of February-April 2025 as a project for their third-year Queen's University School of Computing course: CISC 352 - Artificial Intelligence.

**Usage instructions:**
- Run the TreatmentFlow_Lite.py file for access to our streamlined text-based UI for interacting with all 3 components of TreatmentFlow. Simulate a hospital with inputted patient and bed numbers.
  - This first runs the Deep Learning component from the DeepLearningMLP file to: 
    - Generate the confusion matrix
    - View the model accuracy
  - Then the Constraint Satisfaction component runs from the MIPBedAssignment file to compare the solutions between the MIP and Greedy approaches
    - To check Greedy solutions alone, follow the following steps:
      1. construct a hospital using the HosptialRecords class with a number of beds above 0
      2. generate a list of patients using the gen_patient_list method with a number of patients above 0
      3. construct a scheduler using the Schedule class and the hospital object
      4. use the method(s) run_hospital() (and waiting_times()) to display the Greedy approach's output
  - Finally the Bayesian Network component runs from the BayesSKlearn, which includes:
    - A generator for the Bayesian Network that will be used to predict future diagnoses
    - A document producer that will take a set of observed symptoms as input, and output the most likely diagnoses along with their symptoms

# Installation Instructions

Welcome to our course project! To get our code up and running, you must install pandas, pyreadr, Scikit Learn, and Google's OR Tools.
```
python -m pip install ortools
```
```
pip install pandas pyreadr
```
```
pip install pandas
```
```
python -m pip install -U pip
python -m pip install -U matplotlib
```
```
pip install seaborn
```
```
python -m pip install scikit-learn

```
```
pip install pymc3
```
