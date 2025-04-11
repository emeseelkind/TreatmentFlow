"""
TreatmentFlow_Lite

Authors - Adam Neto and Emese Elkind
April 2025

Text-based UI for interacting with all 3 components of TreatmentFlow:
- Constraint Satisfaction Problem - hospital bed assignment tool
- Deep Learning                   - patient priority classifier
- Bayesian Netowrk                - patient disease document generator
"""

from DeepLearning import DeepLearning as dl
import os
import numpy as np
import pandas as pd
import random


class PatientDatabase:

    def __init__(self, num_patients):
        self.num_patients = num_patients
        self.patient_db = []
        self.user_id = 0
        
        # load pandas patient directory
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
        patient["priority"] = int(0) # MUST USE DL TO PREDICT SYMPTOMS
        patient["symptoms"] = random_row

        self.patient_db.append(patient) 

    def fill_db(self):

        # sample patients from database
        sample_indices = np.random.choice(range(len(self.df)), size=self.num_patients-1, replace=False)        
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
            patient["priority"] = int(ctas)
            patient["symptoms"] = row

            self.patient_db.append(patient) 


class Menu:

    def __init__(self):
        self.num_patients = 0
        self.num_beds = 0
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
        self.num_patients = self.select_int("How many patients should be in the hospital?", 0, 7000)
        self.num_beds = self.select_int("How many beds should be in the hospital?", 0, 1000)

        # building database
        self.patient_data = PatientDatabase(self.num_patients)

        # enter patient info (user is always patient ID 0)
        user_arrival = self.select_int("User arrival time (by the minute)", 0, 1439)
        self.patient_data.add_user_info(user_arrival)
        
        # ** final product should allow users to perform triage survey

        # randomly sampling existing patient profile
        print("Randomly filling patient profiles...")
        self.patient_data.fill_db()

        print("Printing patient db:")
        for row in self.patient_data.patient_db:
            print()
            print(row["id"])
            print(row["priority"])
            print(row["arrival"])

            # for symptom_name, symptom_value in row["symptoms"].items():
            #     print(f"{symptom_name} : {symptom_value}")


my_menu = Menu()
my_menu.run_menu()