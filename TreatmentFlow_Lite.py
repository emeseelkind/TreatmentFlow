"""
TreatmentFlow_Lite

Authors - Adam Neto and Emese Elkind
April 2025

Text-based UI for interacting with all 3 components of TreatmentFlow:
- Constraint Satisfaction Problem - hospital bed assignment tool
- Deep Learning                   - patient priority classifier
- Bayesian Netowrk                - patient disease document generator
"""

from CSP.HospitalClasses import Patient, HospitalRecords
from CSP.GreedyBedAssignment import Scheduler
from DeepLearning import DeepLearning as dl
import os
import numpy as np
import random
import numbers


class PatientDatabase:

    def __init__(self, NUM_PATIENTS, NUM_BEDS):
        self.NUM_PATIENTS = NUM_PATIENTS
        self.NUM_BEDS = NUM_BEDS

        # store patient and hospital data
        self.hospital = HospitalRecords(NUM_BEDS)
        self.patient_db = []
        
        self.user_id = 0
        
        # load pandas patient directory (from CTAS spreadsheets)
        current_dir = os.path.dirname(__file__)
        patient_samples_dir = os.path.join(current_dir, "CTAS_files")
        self.df = dl.load_data_printless(patient_samples_dir)

    def add_user_info(self, arrival):

        print("Randomizing user symptoms...")
        random_index = np.random.choice(len(self.df))
        random_row = self.df.iloc[random_index].drop("esi")

        self.user_id = len(self.patient_db)

        patient = {}
        patient["id"] = self.user_id
        patient["arrival"] = arrival
        patient["ctas"] = int(0) # ** MUST USE DL TO PREDICT SYMPTOMS
        patient["symptoms"] = random_row

        self.patient_db.append(patient) 

    def fill_db(self):

        # sample patients from database
        sample_indices = np.random.choice(range(len(self.df)), size=self.NUM_PATIENTS-1, replace=False)        
        sample_df = self.df.iloc[sample_indices]
        ctas_values = sample_df["esi"]
        sample_patients = sample_df.drop('esi', axis=1)
        # sample_patients = self.df.drop('esi', axis=1).iloc[sample_indices]             

        i = 0
        for ctas, row in zip(ctas_values, sample_patients.to_dict(orient='records')):
            i += 1

            patient = {}
            patient["id"] = i
            patient["arrival"] = random.randint(0, 1439)
            patient["ctas"] = int(ctas)
            patient["symptoms"] = row

            self.patient_db.append(patient)

    def assign_beds(self, printing=False):

        # add patients from patients db to hospital db
        for patient in self.patient_db:
            
            current_patient = Patient()
            current_patient.fill_patient_stats(patient["id"], patient["arrival"], patient["ctas"])
            self.hospital.patient_list.append(current_patient)

        Scheduler(self.hospital).run_hospital(printing)


class Menu:

    def __init__(self):
        self.NUM_PATIENTS = 0
        self.NUM_BEDS = 0
        self.patient_db = None

    def select_int(self, message, lb, ub):
        # guarantees proper user input
        output = input(message + ": ")

        while True:
            try:
                output = int(output)
                if lb <= output <= ub:
                    return output
                else:
                    output = input(f"Input must be within bounds ({lb}, {ub}): ")
            except ValueError:
                output = input("Input must be an integer: ")

    def run_menu(self):

        print("Welcome to TreatmentFlow Lite!")

        # general statistics inquiry
        self.NUM_PATIENTS = self.select_int("How many patients should be in the hospital?", 0, 7000)
        self.NUM_BEDS = self.select_int("How many beds should be in the hospital?", 0, 1000)

        # building database
        self.patient_data = PatientDatabase(self.NUM_PATIENTS, self.NUM_BEDS)

        # enter patient info (user is always patient ID 0)
        user_arrival = self.select_int("User arrival time (by the minute)", 0, 1439)
        self.patient_data.add_user_info(user_arrival)
        
        # ** final product should allow users to perform triage survey

        # randomly sampling existing patient profile
        print("Randomly filling patient profiles...")
        self.patient_data.fill_db()

        print("Printing patient db:") # TEMPORARY - FOR TESTING/SHOWCASE
        for row in self.patient_data.patient_db:
            print()
            print(row["id"])
            print(row["ctas"])
            print(row["arrival"])

            for symptom_name, symptom_value in row["symptoms"].items():
                # print(f"Symptoms: {symptom_name} - {symptom_value}")
                if isinstance(symptom_value, numbers.Number):
                    if symptom_value > 0:
                        print(f"Symptoms: {symptom_name} - {symptom_value}")

        # assign patients to beds
        self.patient_data.assign_beds(True)


my_menu = Menu()
my_menu.run_menu()