# TreatmentFlow

TreatmentFlow is an AI-based project for optimizing the flow of patients through the emergency room.
It includes 3 major components:
- Deep Learning-based patient priority assignment through an automated triage system
- Constraint Optimization-based hospital bed assignment based on incoming patients and available resources
- Bayesian Network-based diagnostic tools to produce bedside documents for doctors only, which display the probabilities of certain conditions based on symptom inputs

TreatmentFlow was originally created by Adam Neto and Emese Elkind during the months of February-April 2025 as a project for their third-year Queen's University School of Computing course: CISC 352 - Artificial Intelligence.

Usage instructions:
- For Constraint Satisfaction components, run the MIPBedAssignment file to compare the solutions between the MIP and Greedy approaches
  - To check Greedy solutions alone, follow the following steps:
    1. construct a hospital using the HosptialRecords class with a number of beds above 0
    2. generate a list of patients using the gen_patient_list method with a number of patients above 0
    3. construct a scheduler using the Schedule class and the hospital object
    4. use the method(s) run_hospital() (and waiting_times()) to display the Greedy approach's output
- For Deep Learning components...

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
