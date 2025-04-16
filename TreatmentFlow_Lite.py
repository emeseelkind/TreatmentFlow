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
from DeepLearning import DLRaw as dl
from BayesNets import BNRaw as bn
import os
import numpy as np
import random
import numbers


class PatientDatabase:

    def __init__(self, num_patients, num_beds):

        # create dl model
        self.dl = dl.DeepLearningTriage()

        # create bn model
        self.bn = bn.BayesNetsDiagnostics()

        self.num_patients = num_patients
        
        # create databases
        self.update_hospital(num_beds)
        self.patient_db = []
        
        # note that user values are not yet set
        self.user_id = -1

    def update_hospital(self, num_beds):

        # set number of beds
        self.num_beds = num_beds

        # create hosptial objects
        self.hospital = HospitalRecords(num_beds)
        self.scheduler = Scheduler(self.hospital)

    def add_user_info(self, arrival, symptoms=None):

        patient = {}
        patient["id"] = 0
        patient["arrival"] = arrival

        print("Predicting CTAS value with DL...")

        # return randomized patient stats, predict CTAS value with DL
        user_symptoms, patient["ctas"] = self.dl.predict_esi(symptoms)
        patient["symptoms"] = user_symptoms.to_dict(orient='records')[0]

        # add to local database
        if self.user_id < 0: # must replace index 0 if user exists
            self.patient_db.insert(0, patient)
            self.user_id = 0
        else:
            self.patient_db[0] = patient

        # mark user as already set
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
        sample_indices = np.random.choice(range(len(self.dl.df)), size=self.num_patients-1, replace=False)
        sample_df = self.dl.df.iloc[sample_indices]
        ctas_values = sample_df["esi"]
        sample_patients = sample_df.drop('esi', axis=1)

        i = 0
        for ctas, row in zip(ctas_values, sample_patients.to_dict(orient='records')):
            i += 1

            patient = {}
            patient["id"] = i
            patient["arrival"] = random.randint(0, 1439)
            patient["ctas"] = int(ctas)
            patient["symptoms"] = row

            self.patient_db.append(patient)

    def assign_beds(self):

        # MUST ONLY BE CALLED WHEN A CHANGE IS MADE - will randomize the service times of each patient

        # reset patient list in hospital object
        self.hospital.patient_list = []

        # add patients from patients db to hospital object db
        for patient in self.patient_db:
            
            # set patient object
            current_patient = Patient()
            current_patient.fill_patient_stats(patient["id"], patient["arrival"], patient["ctas"])

            # insert patient object
            patient["object"] = current_patient
            self.hospital.patient_list.append(current_patient)

    def assign_bed_to_user(self):

        # special function to only adjust the user's object status, not other patients

        # don't assign user if they are not set
        if self.user_id == -1:
            print("User cannot be assigned bed before being uploaded")
            return
        
        # access patient dict
        patient = self.patient_db[0]

        # set patient object
        current_patient = Patient()
        current_patient.fill_patient_stats(patient["id"], patient["arrival"], patient["ctas"])

        # insert patient object
        patient["object"] = current_patient
        self.hospital.patient_list[0] = current_patient
 
    def run_hosp(self, printing):

        self.hospital.reset_service()
        self.scheduler.run_hospital(printing)


class Menu:

    def __init__(self):

        self.num_patients = 0
        self.num_beds = 0
        self.patient_db = None

    def select_int(self, message, lb, ub):

        # bypass query if only one option
        if lb == ub: # assume proper input (no ub < lb)
            return lb

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

    def triage_survey(self):

        # store user triage information from survey
        user = {}        

        # minimum and maximum values set by full dataset min/max values

        print("\nTriage survey:")

        # introductory questions (age, sex)
        user["age"] = self.select_int("Age", 0, 150)
        sex_num = self.select_int("Sex (M=1, F=2)", 1, 2)
        if sex_num == 1:
            user["gender"] = "Male"
        else:
            user["gender"] = "Female"
        
        # vitals
        print("\nVitals:")
        user["triage_vital_sbp"] = self.select_int("Systolic blood pressure", 45, 312)
        user["triage_vital_dbp"] = self.select_int("Diastolic blood pressure", 25, 214)
        user["triage_vital_hr"] = self.select_int("Heart rate", 30, 280)
        user["triage_vital_temp"] = self.select_int("Body temperature (*F)", 90, 106)
        user["triage_vital_o2"] = self.select_int("Oxygen saturation (%)", 60, 99)

        # systems
        print("\nBodily systems (all answers y=1 or n=0):")

        # psycho-social
        user["cc_anxiety"] = self.select_int("Anxiety", 0, 1)
        user["cc_agitation"] = self.select_int("Agitiation", 0, 1)
        user["cc_blurredvision"] = self.select_int("Blurred vision", 0, 1)
        user["cc_fatigue"] = self.select_int("Fatigue", 0, 1)

        # cardiovascular
        user["cc_chestpain"] = self.select_int("Chest pain", 0, 1)

        swelling = self.select_int("Experiencing swelling", 0, 1) # only ask about specific swelling if necessary
        user["cc_legswelling"] = self.select_int("Legs swelling", 0, swelling)
        user["cc_armswelling"] = self.select_int("Arms swelling", 0, swelling)
        user["cc_facialswelling"] = self.select_int("Facial swelling", 0, swelling)
        user["cc_fingerswelling"] = self.select_int("Finger swelling", 0, swelling)
        user["cc_jointswelling"] = self.select_int("Joint swelling", 0, swelling)
        
        # respiratory
        user["cc_shortnessofbreath"] = self.select_int("Shortness of breath", 0, 1)
        user["cc_dyspnea"] = user["cc_shortnessofbreath"]
        user["cc_cough"] = self.select_int("Cough", 0, 1)

        # gastrointestinal
        user["cc_nausea"] = self.select_int("Nausea", 0, 1)
        user["cc_emesis"] = self.select_int("Vomiting", 0, 1)
        user["cc_diarrhea"] = self.select_int("Diarrhea", 0, 1)
        
        # genitourinary
        user["cc_dysuria"] = self.select_int("Difficulty or pain urinating", 0, 1)
        
        # skin
        skin_issues = self.select_int("Experiencing skin issues", 0, 1)
        user["cc_rash"] = self.select_int("Skin rash", 0, skin_issues)
        user["cc_skinirritation"] = self.select_int("Skin irritation", 0, skin_issues)
        user["ulcerskin"] = self.select_int("Ulcers", 0, skin_issues)        
        user["cc_skinproblem"] = self.select_int("Other skin problems", 0, skin_issues)

        # musculoskeletal/pain

        # PAIN IN REGIONS (only ask questions if user experiencing pain in region)

        # master attribute, determines whether others can be true
        user["cc_pain"] = self.select_int("\nExperiencing pain", 0, 1)

        # head/face/neck
        hfn_pain = self.select_int("Pain in the head/face/neck", 0, user["cc_pain"])
        user["cc_headache"] = self.select_int("Headache", 0, hfn_pain)
        user["cc_headpain"] = self.select_int("Head pain", 0, hfn_pain)
        user["cc_dentalpain"] = self.select_int("Tooth pain", 0, hfn_pain)
        user["cc_earpain"] = self.select_int("Ear pain", 0, hfn_pain)
        user["cc_eyepain"] = self.select_int("Eye pain", 0, hfn_pain)
        user["cc_facialpain"] = self.select_int("Facial pain", 0, hfn_pain)
        user["cc_jawpain"] = self.select_int("Jaw pain", 0, hfn_pain)
        user["cc_neckpain"] = self.select_int("Neck pain", 0, hfn_pain)
        
        # chest/thorax
        ct_pain = self.select_int("Pain in the chest/thorax", 0, user["cc_pain"])
        user["cc_ribpain"] = self.select_int("Rib pain", 0, ct_pain)
        user["cc_breastpain"] = self.select_int("Breast pain", 0, ct_pain)

        # back
        user["cc_backpain"] = self.select_int("Back pain", 0, user["cc_pain"])

        # abdomen/GI
        agi_pain = self.select_int("Pain in the abdomen/gastrointestinal region", 0, user["cc_pain"])
        user["cc_abdominalpain"] = self.select_int("Abdominal pain", 0, agi_pain)
        if sex_num == 2:
            user["cc_abdominalpainpregnant"] = self.select_int("Abdominal pain related to pregnancy", 0, agi_pain)
        else:
            user["cc_abdominalpainpregnant"] = 0
        user["cc_epigastricpain"] = self.select_int("Upper abdomen pain", 0, agi_pain)
        user["cc_flankpain"] = self.select_int("Side of abdomen pain", 0, agi_pain)
        user["cc_pelvicpain"] = self.select_int("Pelvic pain", 0, agi_pain)
        user["cc_rectalpain"] = self.select_int("Rectal pain", 0, agi_pain)

        # genitals
        gen_pain = self.select_int("Pain of the genitals", 0, user["cc_pain"])
        if sex_num == 1: # if biologically male
            user["cc_testiclepain"] = self.select_int("Testicle pain", 0, gen_pain)
            user["cc_vaginalpain"] = 0
        else: # if biologically female
            user["cc_testiclepain"] = 0
            user["cc_vaginalpain"] = self.select_int("Vaginal pain", 0, gen_pain)
        user["cc_groinpain"] = self.select_int("Groin pain", 0, gen_pain)

        # lower limbs
        lower_pain = self.select_int("Pain of the lower limbs", 0, user["cc_pain"])
        user["cc_legpain"] = self.select_int("Leg pain", 0, lower_pain)
        user["cc_kneepain"] = self.select_int("Knee pain", 0, lower_pain)
        user["cc_anklepain"] = self.select_int("Ankle pain", 0, lower_pain)
        user["cc_toepain"] = self.select_int("Toe pain", 0, lower_pain)
        user["cc_footpain"] = self.select_int("Foot pain", 0, lower_pain)
        user["cc_hippain"] = self.select_int("Hip pain", 0, lower_pain)

        # upper limbs
        upper_pain = self.select_int("Pain of the upper limbs", 0, user["cc_pain"])
        user["cc_armpain"] = self.select_int("Arm pain", 0, upper_pain)
        user["cc_handpain"] = self.select_int("Hand pain", 0, upper_pain)
        user["cc_fingerpain"] = self.select_int("Finger pain", 0, upper_pain)
        user["cc_wristpain"] = self.select_int("Wrist pain", 0, upper_pain)
        user["cc_elbowpain"] = self.select_int("Elbow pain", 0, upper_pain)
        user["cc_shoulderpain"] = self.select_int("Shoulder pain", 0, upper_pain)


        # RECENT INJURY IN REGIONS (only ask questions if user experiencing pain in region)
        injury = self.select_int("\nRecent injury", 0, 1)

        # head/face/neck
        hfn_injury = self.select_int("Injury to the head/face/neck", 0, injury)
        user["cc_headinjury"] = self.select_int("Head injury", 0, hfn_injury)
        user["cc_facialinjury"] = self.select_int("Facial injury", 0, hfn_injury)
        user["cc_eyeinjury"] = self.select_int("Eye injury", 0, hfn_injury)
        
        # chest/rib/thorax
        user["cc_ribinjury"] = self.select_int("Rib injury", 0, injury)

        # lower limbs
        lower_injury = self.select_int("Injury to the lower limbs", 0, injury)
        user["cc_leginjury"] = self.select_int("Leg injury", 0, lower_injury)
        user["cc_kneeinjury"] = self.select_int("Knee injury", 0, lower_injury)
        user["cc_footinjury"] = self.select_int("Foot injury", 0, lower_injury)
        user["cc_toeinjury"] = self.select_int("Toe injury", 0, lower_injury)
        user["cc_ankleinjury"] = self.select_int("Ankle injury", 0, lower_injury)

        # upper limbs
        upper_injury = self.select_int("Injury to the upper limbs", 0, injury)
        user["cc_arminjury"] = self.select_int("Arm injury", 0, upper_injury)
        user["cc_handinjury"] = self.select_int("Hand injury", 0, upper_injury)
        user["cc_fingerinjury"] = self.select_int("Finger injury", 0, upper_injury)
        user["cc_thumbinjury"] = self.select_int("Thumb injury", 0, upper_injury)
        user["cc_wristinjury"] = self.select_int("Wrist injury", 0, upper_injury)
        user["cc_shoulderinjury"] = self.select_int("Shoulder injury", 0, upper_injury)


        # LACERATION/CUTS
        user["cc_laceration"] = self.select_int("Laceration (cuts)", 0, 1)
        user["cc_faciallaceration"] = self.select_int("Facial laceration", 0, user["cc_laceration"])
        user["cc_headlaceration"] = self.select_int("Head laceration", 0, user["cc_laceration"])
        user["cc_extremitylaceration"] = self.select_int("Extremity laceration", 0, user["cc_laceration"])

        # return symptom dictionary
        return user

    def observe_patient(self, patient_id):

        # set patient_dict to appropriate capsule of information
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
                    # produce diagnostic informaiton using Bayes Nets module
                    self.patient_data.bn.diagnostic_doc_ext(patient_id, patient_dict["symptoms"])

                case 4:
                    return

    def access_patients(self):
        
        self.patient_data.run_hosp(False)

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

                    # randomize symptoms if preferred
                    if self.select_int("Perform triage", 0, 1):
                        self.patient_data.add_user_info(user_arrival, self.triage_survey())
                    else:
                        self.patient_data.add_user_info(user_arrival)

                    # only update user's object info, don't reset other service times
                    self.patient_data.assign_bed_to_user()

                case 2:
                    self.num_beds = self.select_int("Number of beds", 1, 1000)
                    self.patient_data.update_hospital(self.num_beds)

                    # ensure patients are appropriately assigned
                    self.patient_data.assign_beds()

                case 3:
                    self.num_patients = self.select_int("Number of patients", 1, 7000)
                    self.patient_data.num_patients = self.num_patients

                    # randomly sampling existing patient profile
                    self.patient_data.fill_db()

                    # ensure patients are appropriately assigned
                    self.patient_data.assign_beds()

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

        # assign hospital beds
        self.patient_data.assign_beds()

        while True:

            # core menu
            print("\nPlease select an option: ")
            print(" 1. Access patient database")
            print(" 2. Print bed assignments")
            print(" 3. Update hospital database")
            print(" 4. Quit")

            response = self.select_int("Choice", 1, 4)
            match response:
                case 1:
                    # access patient menu
                    self.access_patients()

                case 2:
                    # print bed assignment updates
                    self.patient_data.run_hosp(True)

                case 3:
                    # access updates menu
                    self.update_stats()

                case 4:
                    # end services
                    return


my_menu = Menu()
my_menu.run_menu()
