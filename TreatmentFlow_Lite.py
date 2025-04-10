"""
TreatmentFlow_Lite

Authors - Adam Neto and Emese Elkind
April 2025

Text-based UI for interacting with all 3 components of TreatmentFlow:
- Constraint Satisfaction Problem - hospital bed assignment tool
- Deep Learning                   - patient priority classifier
- Bayesian Netowrk                - patient disease document generator
"""

import random

class PatientDatabase:

    def __init__(self, num_patients):
        self.num_patients = num_patients
        self.patient_db = [{"id": 0, "arrival": 0, "symptoms": "SYMPTOMS FROM USER"}]

    def fill_db(self):
        for i in range(1, self.num_patients):
            patient = {}
            patient["id"] = i
            patient["arrival"] = random.randint(0, 1439)
            patient["symptoms"] = "REST OF SYMPTOMS FROM PATIENT"
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
        self.patient_db = PatientDatabase(self.num_patients)

        # enter patient info (user is always patient ID 0)
        self.patient_db[0]["arrival"] = self.select_int("User arrival time (by the minute): ", 0, 1439)

        # final product should allow users to perform triage survey
        print("Randomizing symptoms for user...")        

        # randomly sampling existing patient profile
        print("Randomly filling patient profiles...")
        self.patient_db.fill_db()




my_menu = Menu()
my_menu.run_menu()