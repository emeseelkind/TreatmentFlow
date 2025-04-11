"""
TreatmentFlow_Lite

Authors - Adam Neto and Emese Elkind
April 2025

Text-based UI for interacting with all 3 components of TreatmentFlow:
- Constraint Satisfaction Problem - hospital bed assignment tool
- Deep Learning                   - patient priority classifier
- Bayesian Netowrk                - patient disease document generator
"""

from CSP.HospitalClasses import Patient, HospitalRecords, print_time
from CSP.GreedyBedAssignment import Scheduler
from DeepLearning import DeepLearning as dl
import os
import numpy as np
import random
import numbers


class PatientDatabase:

    def __init__(self, num_patients, num_beds):
        self.num_patients = num_patients
        
        # create databases
        self.update_hospital(num_beds)
        self.patient_db = []
        
        # note that user values are not yet set
        self.user_id = -1
        
        # load pandas patient directory (from CTAS spreadsheets)
        current_dir = os.path.dirname(__file__)
        patient_samples_dir = os.path.join(current_dir, "CTAS_files")
        self.df = dl.load_data_printless(patient_samples_dir)

    def update_hospital(self, num_beds):
        # set number of beds
        self.num_beds = num_beds

        # create hosptial objects
        self.hospital = HospitalRecords(num_beds)
        self.scheduler = Scheduler(self.hospital)

    def add_user_info(self, arrival):

        print("Randomizing user symptoms...")
        random_index = np.random.choice(len(self.df))
        random_row = self.df.iloc[random_index].drop("esi")

        self.user_id = 0

        patient = {}
        patient["id"] = self.user_id
        patient["arrival"] = arrival
        patient["ctas"] = int(0) # ** MUST USE DL TO PREDICT SYMPTOMS
        patient["symptoms"] = random_row

        self.patient_db.insert(0, patient)
        self.user_id = 0

    def fill_db(self):

        # reset database
        if self.user_id < 0:
            self.patient_db = []
        else:
            temp = self.patient_db[0]
            self.patient_db = []
            self.patient_db.append(temp)

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
            patient["ctas"] = int(ctas)
            patient["symptoms"] = row

            self.patient_db.append(patient)

    def assign_beds(self, printing=False):

        # add patients from patients db to hospital db
        for patient in self.patient_db:
            
            current_patient = Patient()
            current_patient.fill_patient_stats(patient["id"], patient["arrival"], patient["ctas"])

            patient["object"] = current_patient
            self.hospital.patient_list.append(current_patient)

        self.scheduler.run_hospital(printing)


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

    def observe_patient(self, patient_id):
        if self.patient_data.user_id < 0:
            patient_dict = self.patient_data.patient_db[patient_id - 1]
        else:
            patient_dict = self.patient_data.patient_db[patient_id]

        while True:
            # core menu
            print(f"\nPatient {patient_id}. CTAS {patient_dict['ctas']}: ")
            print(" 1. Service times")
            print(" 2. Symptom list")
            print(" 3. Patient document")
            print(" 4. Exit")

            response = self.select_int("Choice", 1, 4)
            match response:
                case 1:
                    print(f"\nPatient {patient_id} service time info:")
                    
                    # print out patient arrival, service time, discharge time
                    print(f"Arrival: {print_time(patient_dict['arrival'])}")
                    print(f"Service time: {print_time(patient_dict['object'].service_time)}")

                    if patient_dict["object"].service_start < 0:
                        print(f"Patient {patient_id} was never served.")
                    else:
                        print(f"Waiting time: {print_time(patient_dict['object'].get_waiting_time())}")
                        print(f"First served: {print_time(patient_dict['object'].service_start)}")

                case 2:
                    print(f"\nPatient {patient_id} symptom info:")

                    # print out patient symptoms
                    for symptom_name, symptom_value in patient_dict["symptoms"].items():
                        if isinstance(symptom_value, numbers.Number):
                            if symptom_value > 0:
                                print(f"{symptom_name}: {symptom_value}")

                case 3:
                    print(f"PLACEHOLDER: Bayes output for patient {patient_id}")

                case 4:
                    return

    def access_patients(self):
        
        self.patient_data.assign_beds()

        while True:

            # core menu
            print("\nPatient database: ")
            print(" 1. Select user")
            print(" 2. Select other patient")
            print(" 3. Exit")

            response = self.select_int("Choice", 1, 3)
            match response:
                case 1:
                    if self.patient_data.user_id < 0:
                        print("User data must be uploaded before access.")
                    else:
                        self.observe_patient(0)

                case 2:
                    # prevent list overflow errors
                    min_required = 0 if self.patient_data.user_id < 0 else 1
                    
                    if len(self.patient_data.patient_db) > min_required:
                        self.observe_patient(self.select_int("Select patient ID", 1, self.num_patients - 1))
                    else:
                        print("Additional patient data must be uploaded before access.")

                case 3:
                    return

    def update_stats(self):

        while True:

            # core menu
            print("\nHospital stats: ")
            print(f" Beds: {self.num_beds}")
            print(f" Patients: {self.num_patients}")

            print("\nUpdate options: ")
            print(" 1. Update user info")
            print(" 2. Update hospital size")
            print(" 3. Update patient list")
            print(" 4. Exit")

            response = self.select_int("Choice", 1, 4)
            match response:
                case 1:
                    # enter user patient info (user is always patient ID 0)
                    user_arrival = self.select_int("User arrival time (by the minute)", 0, 1439)
                    self.patient_data.add_user_info(user_arrival)

                    # ** final product should allow users to perform triage survey
                case 2:
                    num_beds = self.select_int("Number of beds", 1, 1000)
                    self.patient_data.update_hospital(num_beds)

                case 3:
                    self.num_patients = self.select_int("Number of patients", 1, 7000)
                    self.patient_data.num_patients = self.num_patients

                    # randomly sampling existing patient profile
                    self.patient_data.fill_db()

                case 4:
                    return

    def run_menu(self):

        print("\nWelcome to TreatmentFlow Lite!")

        # initial setup
        self.num_patients = self.select_int("How many patients should be in the hospital?", 1, 7000)
        self.num_beds = self.select_int("How many beds should be in the hospital?", 1, 1000)

        # building database
        self.patient_data = PatientDatabase(self.num_patients, self.num_beds)

        # enter patient info (user is always patient ID 0)
        user_arrival = self.select_int("User arrival time (by the minute)", 0, 1439)
        self.patient_data.add_user_info(user_arrival)
        self.patient_data.fill_db()

        while True:

            # core menu
            print("\nPlease select an option: ")
            print(" 1. Access patient database")
            print(" 2. Print bed assignments")
            print(" 3. Update hospital database")
            print(" 4. Quit")

            response = self.select_int("Choice", 1, 6)
            match response:
                case 1:
                    self.access_patients()

                case 2:
                    # print bed assignment updates
                    self.patient_data.assign_beds(True)

                case 3:
                    self.update_stats()

                case 4:
                    return


my_menu = Menu()
my_menu.run_menu()