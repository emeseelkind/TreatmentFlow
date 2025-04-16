"""
TreatmentFlow
Bayesian Networks - Disease Diagnostics from Triage Inputs (Printless Raw Version for integration)

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from datetime import datetime
import os

class BayesNetsDiagnostics:

    def __init__(self):
        
        # load data from source db (symbipredict)
        file_path = os.path.join(os.path.dirname(__file__), "symbipredict_2022.csv")
        self.df = self.load_data(file_path)

        # process data before building model
        X, y = self.preprocess_data(self.df)

        # TRAIN MODEL ON 100% OF THE DATA, NOT 80%

        # build Bayesian Network
        self.model = self.train_model(X, y)

    def load_data(self, file_path):
        """
        Load data from a CSV file and return it as a pandas DataFrame.
        
        Parameters:
        file_path : str - Path to the CSV file
            
        Returns:
        pd.DataFrame - Loaded data as a DataFrame
        """
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        """
        Preprocess the data by preparing features and target.
        
        Parameters:
        data : pd.DataFrame - DataFrame containing the dataset
            
        Returns:
        tuple- X (features DataFrame), y (target Series)
        """
        # features (X) and target (y)
        X = data.drop('prognosis', axis=1)
        y = data['prognosis']
        
        return X, y

    def train_model(self, X_train, y_train, alpha=1.0):
        """
        Train a Bernoulli Naive Bayes model.
        
        Parameters:
        X_train : pd.DataFrame -Training features
        y_train : pd.Series - Training target
        alpha : float - Laplace/Lidstone smoothing parameter
            
        Returns:
        model : BernoulliNB
        """
        # Train Bernoulli Naive Bayes model (appropriate for binary features)
        model = BernoulliNB(alpha=alpha)
        model.fit(X_train, y_train)

        return model

    def predict_disease(self, model, symptoms_dict, X_columns):
        """
        Predict disease probabilities based on provided symptoms
        
        Parameters:
        model : BernoulliNB
        symptoms_dict : dict - Dictionary with symptom names as keys and 0/1 as values
        X_columns : list - List of all possible symptom names
            
        Returns:
        dict - Disease names and their probabilities, sorted by probability
        """
        # DataFrame with all symptoms set to 0
        patient = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
        
        # Set the provided symptoms to 1
        for symptom, value in symptoms_dict.items():
            if symptom in patient.columns:
                patient[symptom] = value
            else:
                print(f"Warning: Symptom '{symptom}' not recognized and will be ignored")
        
        probs = model.predict_proba(patient)[0]
        
        # Map probabilities to disease names and sort
        disease_probs = dict(zip(model.classes_, probs))
        sorted_disease_probs = {k: v for k, v in sorted(disease_probs.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_disease_probs

    def diagnose_patient(self, model, symptoms_dict, X_columns, top_n=5):
        """
        Diagnose a patient based on reported symptoms
        
        Parameters:
        model : BernoulliNB
        symptoms_dict : Dictionary with symptom names as keys and 0/1 as values
        X_columns : list of all possible symptom names
        top_n : int - Number of top diagnoses to return
            
        Returns:
        List of tuples (disease, probability)
        """
        disease_probs = self.predict_disease(model, symptoms_dict, X_columns)
        
        # Get top 5 diseases with highest probabilities
        top_diseases = list(disease_probs.items())[:top_n]
        
        return top_diseases

    def important_symptoms_for_disease(self, model, disease_name, X_columns, top_n=10):
        """
        Identify the most important symptoms for a given disease based on feature probabilities
        
        Parameters:
        model : BernoulliNB
        disease_name : str - Name of the disease
        X_columns : list- List of all possible symptom names
        top_n : int - Number of top symptoms to return
            
        Returns:
        dict - Symptom names and their importance scores
        """
        if disease_name not in model.classes_:
            return f"Disease '{disease_name}' not found in the model's classes"
        
        # Get the disease index
        disease_idx = np.where(model.classes_ == disease_name)[0][0]
        feature_probs = model.feature_log_prob_[disease_idx]
        
        # Create a dictionary of symptom names and their importance
        importance = {}
        for i, symptom in enumerate(X_columns):
            # Convert log probability back to probability and use the difference
            # between P(symptom=1|disease) and P(symptom=1) as importance
            prob_symptom_given_disease = np.exp(feature_probs[i])
            importance[symptom] = prob_symptom_given_disease
        
        # Sort and get top N symptoms
        sorted_importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        
        return dict(list(sorted_importance.items())[:top_n])

    def generate_bedside_document(self, patient_id, symptoms_dict, top_diagnoses, model, X_columns, output_dir="bedside_documents"):
        """
        Generate a doctor-focused bedside document with diagnosis probabilities
        
        Parameters:
        patient_id : str - Patient identifier
        symptoms_dict : Dictionary with symptom names as keys and 0/1 as values
        top_diagnoses : List of tuples (disease, probability)
        model : BernoulliNB
        X_columns : List of all possible symptom names
        output_dir : str -Directory to save the document
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/diagnosis_{patient_id}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"CLINICAL DIAGNOSTIC ASSESSMENT - CONFIDENTIAL\n")
            f.write(f"PATIENT ID: {patient_id}\n")
            f.write(f"DATE/TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Reported Symptoms
            f.write("REPORTED SYMPTOMS:\n")
            f.write("-"*80 + "\n")
            positive_symptoms = [s.replace('_', ' ') for s, v in symptoms_dict.items() if v == 1]
            if positive_symptoms:
                for symptom in positive_symptoms:
                    f.write(f"+ {symptom}\n")
            else:
                f.write("No positive symptoms reported\n")
            f.write("\n")
            
            # Diagnosis Probabilities
            f.write("DIFFERENTIAL DIAGNOSIS:\n")
            f.write("-"*80 + "\n")
            f.write("| {:<30} | {:<15} | {:<30} |\n".format("Diagnosis", "Probability", "Confidence Level"))
            f.write("|" + "-"*32 + "|" + "-"*17 + "|" + "-"*32 + "|\n")
            
            for i, (disease, prob) in enumerate(top_diagnoses):
                # Determine confidence level
                if prob >= 0.8:
                    confidence = "Very High"
                elif prob >= 0.6:
                    confidence = "High"
                elif prob >= 0.4:
                    confidence = "Moderate"
                elif prob >= 0.2:
                    confidence = "Low"
                else:
                    confidence = "Very Low"
                    
                # Format with rank indicator
                rank_indicator = "*" if i == 0 else ""
                f.write("| {:<30} | {:<15.4f} | {:<30} {}\n".format(
                    disease, prob, confidence, rank_indicator
                ))
            
            f.write("\n* Primary diagnosis candidate\n\n")
            
            # Supporting Evidence for Primary Diagnosis
            primary_diagnosis = top_diagnoses[0][0]
            f.write(f"SUPPORTING EVIDENCE FOR PRIMARY DIAGNOSIS ({primary_diagnosis}):\n")
            f.write("-"*80 + "\n")
            
            # Get important symptoms for primary diagnosis
            key_symptoms = self.important_symptoms_for_disease(model, primary_diagnosis, X_columns, top_n=10)
            
            f.write("| {:<30} | {:<15} | {:<30} |\n".format("Symptom", "Importance", "Present in Patient"))
            f.write("|" + "-"*32 + "|" + "-"*17 + "|" + "-"*32 + "|\n")
            
            for symptom, importance in key_symptoms.items():
                present = "Yes" if symptoms_dict.get(symptom, 0) == 1 else "No"
                f.write("| {:<30} | {:<15.4f} | {:<30} |\n".format(
                    symptom.replace('_', ' '), importance, present
                ))
            
            f.write("\n")
            
            # Alternative Diagnoses
            if len(top_diagnoses) > 1:
                f.write("KEY DIFFERENTIATING FACTORS FOR ALTERNATIVE DIAGNOSES:\n")
                f.write("-"*80 + "\n")
                
                for disease, prob in top_diagnoses[1:4]:  # Look at up to 3 alternatives
                    f.write(f"\n{disease} (Probability: {prob:.4f}):\n")
                    alt_symptoms = self.important_symptoms_for_disease(model, disease, X_columns, top_n=5)
                    
                    for symptom, importance in alt_symptoms.items():
                        present = "Yes" if symptoms_dict.get(symptom, 0) == 1 else "No"
                        f.write(f"- {symptom.replace('_', ' ')}: Importance={importance:.4f}, Present={present}\n")
            
            # Additional Notes and Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-"*80 + "\n")
            f.write("1. Consider confirmatory tests for primary diagnosis\n")
            f.write("2. Monitor for development of additional symptoms\n")
            f.write("3. Re-evaluate diagnosis if patient condition changes\n")
            
            # Footer
            f.write("\n" + "="*80 + "\n")
            f.write("NOTICE: This document is generated by an AI diagnostic assistant and\n")
            f.write("is intended for clinical decision support only. Final diagnosis and\n")
            f.write("treatment plans should be determined by qualified healthcare providers.\n")
            f.write("="*80 + "\n")
        
        print(f"Bedside document generated: {filename}")
        
        # Also print a console-friendly version
        print("\nDIAGNOSIS SUMMARY:")
        print("-"*50)
        for i, (disease, prob) in enumerate(top_diagnoses):
            print(f"{i+1}. {disease}: {prob:.4f}")
        
        return filename

    def diagnostic_doc_ext(self, patient_id, patient_dict):
        """
        Takes in a patient from DL pd dataframe format
        Uses translation CSV to get a BN-format patient dict
        Uses this patient dict to invoke the diagnosis doc generator

        Parameters:
        patient_id : ID of patient
        patient_dict : dictionary with patient status from DL db
        """

        # find source for translation csv
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        translation_path = os.path.join(parent_dir, 'BN-DL_translation.csv')

        # create translation dataframe
        translation = pd.read_csv(translation_path)
        translation = translation.drop('prognosis', axis=1) # remove prognosis from df

        possible_symptoms = translation.columns

        # create dict to be used for diagnosis functions
        symptoms_dict = {}

        # TRANSLATE THE FORMAT OF THE PATIENT
        for BN_symptom in translation:
        
            # set default value for dictionary item for this symptom
            symptoms_dict[BN_symptom] = 0

            # low fever, high fever, high hr

            if BN_symptom == "mild_fever":

                # mild fever if body temp between 100 and 103
                symptoms_dict[BN_symptom] = int(100 < patient_dict["triage_vital_temp"] < 103)

            elif BN_symptom == "high_fever":

                # high fever if body temp at least 103
                symptoms_dict[BN_symptom] = int(patient_dict["triage_vital_temp"] >= 103)

            elif BN_symptom == "fast_heart_rate":

                # fast HR if HR exceeds 100
                symptoms_dict[BN_symptom] = int(patient_dict["triage_vital_hr"] >= 100)

            else:

                # set or value to a default of 0
                check_or = 0

                # set BN symptom to true if ANY of the associated DL symptoms are true
                for DL_symptom in translation[BN_symptom].dropna():
                    check_or |= int(patient_dict[DL_symptom])

                symptoms_dict[BN_symptom] = check_or

        # get top diagnoses and generate bedside document by invoking model
        top_diagnoses = self.diagnose_patient(self.model, symptoms_dict, possible_symptoms)
        self.generate_bedside_document(patient_id, symptoms_dict, top_diagnoses, self.model, possible_symptoms)
