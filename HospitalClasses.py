"""
TreatmentFlow
Constraint Optimization: Hospital Bed Assignment - Hospital Classes

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

import random

"""
Summary of approach:
- sort list of patients by arrival time (OR create running list that trickles in by arrival time?)
    - sort patients with identical arrival times by priority
        - assign highest priority patients to beds with greedy methods, such that next available bed
            is assigned to next highest priority patient
            - will need a running list of patients to-be-served
                - if a new patient arrives after current patients with a higher priority,
                    they will be assigned a bed first
                        POTENTIAL ISSUE: how to avoid low-priority starvation

Classes involved:

Bed
-   includes information associated with each bed

Patient
-   includes personal patient information

HospitalRecords
-   holds database for patients, beds in hospital

Scheduler
-   simulates the operation of the hospital over a day given patients, beds
"""

# function to print the time on a 24 hour clock based on minute input
def print_time(my_time):
    hours = 0
    minutes = my_time
    while minutes >= 60:
        hours += 1
        minutes -= 60

    print_min = str(minutes)
    if minutes < 10:
        print_min = f"0{minutes}"

    return f"{hours}:{print_min}"

# Bed class
class Bed:
    def __init__(self, id=0) -> None:
        self.id = id
        self.occupied = False
        self.occupant = None

    def assign(self, patient):
        if not type(patient) == Patient:
            raise TypeError("The value to be assigned to a bed not type Patient")
        
        self.occupied = True
        self.occupant = patient

    def discharge(self):
        patient = self.occupant

        self.occupied = False
        self.occupant = None

        return patient

# Patient class includes information about each individual patient
class Patient:
    def __init__(self, id=0, priority=0, arrival_time=0, service_time=0) -> None:
        self.id = id
        self.priority = priority
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.service_start = -1

    def gen_rand_patient(self):
        self.priority = random.randint(1,5)
        self.arrival_time = random.randint(1,1440)
        self.set_service_time()

    def set_service_time(self):
        if self.priority == 1:
            self.service_time = random.randint(10, 60)
        elif self.priority == 2:
            self.service_time = random.randint(20, 90)
        elif self.priority == 3:
            self.service_time = random.randint(30, 180)
        elif self.priority == 4:
            self.service_time = random.randint(60, 360)
        elif self.priority == 5:
            self.service_time = random.randint(120, 500)

    def set_service_start(self, service_start):
        self.service_start = service_start

    def get_waiting_time(self):
        if self.service_start == -1:
            return 0.1 # signifies that the patient was not served in the day
        return self.service_start - self.arrival_time

    def arrival_time_printed(self):
        return print_time(self.arrival_time)
    
    def service_time_printed(self):
        return print_time(self.service_time) 
     
# HosptialRecords class includes information about every patient in the hospital
class HospitalRecords:
    def __init__(self, NUM_BEDS=0) -> None:
        # patient information
        self.patient_list = []
        self.unserved = []
        self.serving = []
        self.max_patient_id = 0

        # resource information
        self.NUM_BEDS = NUM_BEDS
        self.beds_available = NUM_BEDS

        self.beds = []
        for i in range(NUM_BEDS):
            new_bed = Bed(i)
            self.beds.append(new_bed)

    def gen_patient_list(self, numpatients):
        self.max_patient_id = 0
        plist = []
        
        for i in range(numpatients):
            new_patient = Patient(i)
            new_patient.gen_rand_patient()

            plist.append(new_patient)
            self.insert_unserved(new_patient)

            self.max_patient_id += 1

        self.patient_list = plist

    def add_patient(self, patient):
        if not type(patient) == Patient:
            raise TypeError("The value inserted to the patient_list is not type Patient")
        
        self.max_patient_id += 1
        self.patient_list.append(patient)
        self.insert_unserved(patient)

    def insert_unserved(self, patient):
        if not type(patient) == Patient:
            raise TypeError("The value inserted to the unserved list is not type Patient")
        
        if patient not in self.unserved:
            for i in range(len(self.unserved)):
                if patient.priority > self.unserved[i].priority:
                    self.unserved.insert(i, patient)
                    return
                elif patient.priority == self.unserved[i].priority:
                    if self.unserved[i].arrival_time > patient.arrival_time:
                        self.unserved.insert(i, patient)
                        return
            self.unserved.append(patient)

    def serve_patients(self):
        while self.beds_available and self.unserved:
            self.serve_patient(self.unserved[0])
    
    def serve_patient(self, patient):
        if not type(patient) == Patient:
            raise TypeError("The patient to be served is not type Patient")
        
        if self.beds_available:
            self.unserved.remove(patient)
            self.serving.append(patient)

            for bed in self.beds:
                if not bed.occupied:
                    bed.assign(patient)
                    bed.occupied = True
                    
                    self.beds_available -= 1
                    return

    def discharge_patient(self, patient):
        if not type(patient) == Patient:
            raise TypeError("The patient to be discharged is not type Patient")
        
        self.serving.remove(patient)

        for bed in self.beds:
            if bed.occupant == patient:
                bed.occupant = None
                bed.occupied = False

                self.beds_available += 1
                return
            
# hospital simulator class (for testing / showcase)
class Scheduler:
    def __init__(self, Hospital) -> None:
        if not type(Hospital) == HospitalRecords:
            raise TypeError("The Hospital to be scheduled not type Hospital")
        
        self.Hospital = Hospital

    def run_hospital(self):
        for minute in range(1440):
            change_made = False

            for patient in self.Hospital.serving:
                if minute - patient.service_start >= patient.service_time:
                    self.Hospital.discharge_patient(patient)
                    change_made = True
                    print(f"Discharging {patient.id}")

            for patient in self.Hospital.unserved:
                # print(patient.arrival_time)
                if patient.arrival_time <= minute:
                    if self.Hospital.beds_available:
                        self.Hospital.serve_patient(patient)
                        patient.service_start = minute
                        change_made = True
                        print(f"Serving {patient.id}")
            
            # print current arrangement
            if change_made:
                self.print_arrangement(minute)
                queue = ""
                for patient in self.Hospital.unserved:
                    if patient.arrival_time <= minute:
                        queue = queue + f" {patient.id}:{patient.priority}"
                print(f"Queue: {queue}")
                    
    def waiting_times(self):

        waiting_times_list = []
        for i in range(len(self.Hospital.patient_list)):
            current_patient = self.Hospital.patient_list[i]
            my_waiting_time = current_patient.get_waiting_time()
            
            if my_waiting_time > 0:

                waiting_times_index = 0
                while (waiting_times_index < len(waiting_times_list)) and (my_waiting_time < waiting_times_list[waiting_times_index].get_waiting_time()):
                    waiting_times_index += 1
                
                waiting_times_list.insert(waiting_times_index, current_patient)

        print("\n--Waiting Times--")
        for patient in waiting_times_list:
            print(f"id: {patient.id}, pri: {patient.priority}, wait: {patient.get_waiting_time()}")
                    
    def print_arrangement(self, time):

        WIDTH = 10

        bed_print = []
        row = []

        for bed in self.Hospital.beds:
            if bed.id % WIDTH == 0:
                if bed.id != 0:
                    bed_print.append(row)
                row = []
                
            if bed.occupied:
                row.append(f"{bed.occupant.id}:{bed.occupant.priority}")
            else:
                row.append("   ")

        bed_print.append(row)

        # printing array
        print(f"\n--HOSPITAL AT {print_time(time)}--")
        for this_row in bed_print:
            print(this_row)
