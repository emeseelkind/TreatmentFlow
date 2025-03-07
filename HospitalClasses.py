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
            # self.insert_unserved(new_patient)

            self.max_patient_id += 1

        self.patient_list = plist
        self.unserved = self.mergesort_patient_list(plist)

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

    # run non-time bounded service of all unserved patients (for testing)
    def serve_patients(self):
        while self.beds_available and self.unserved:
            self.serve_patient(self.unserved[0])
    
    # non-time bounded service of patient (for testing)
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
            
    def merge_patient_list(self, left, right):
        """
        Input:
        left: the left list of Patient
        right: the right list of Patient

        Output:
        merged version of left and right patient lists

        Need to merge the lists by priority (descending), arrival time (ascending)
        """

        merged = []

        i = 0
        j = 0
        while i < len(left) and j < len(right):
            l_pri = left[i].priority
            l_ari = left[i].arrival_time

            r_pri = right[j].priority
            r_ari = right[j].arrival_time

            # sort by priority
            if l_pri < r_pri:
                merged.append(right[j])
                j += 1
            elif l_pri > r_pri:
                merged.append(left[i])
                i += 1
            else:
                # if priorities are equal, sort by arrival time
                if l_ari <= r_ari:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1

        # extend end of merged list with rest of left or right
        while i < len(left):
            merged.append(left[i])
            i += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
        
        return merged

    def mergesort_patient_list(self, patient_list):
        """
        Input:
        patients_list: list of Patient

        Output:
        sorted version of patient list by priority (descending), arrival time (ascending)
        """

        # base case
        if len(patient_list) <= 1:
            return patient_list
        elif len(patient_list) == 2:
            # if priority of 0 < priority of 1, switch
            if patient_list[0].priority < patient_list[1].priority:
                patient_list[0], patient_list[1] = patient_list[1], patient_list[0]
            
            # if priority of 0 == priority of 1, look at arrival times
            elif patient_list[0].priority == patient_list[1].priority:
                # if arrival time of 0 > arrival time of 1, switch
                if patient_list[0].arrival_time > patient_list[1].arrival_time:
                    patient_list[0], patient_list[1] = patient_list[1], patient_list[0]
            
            # otherwise, sublist is already sorted
            return patient_list
        
        else:

            # recursive case
            midpoint = len(patient_list) // 2
            left_sorted = self.mergesort_patient_list(patient_list[:midpoint])
            right_sorted = self.mergesort_patient_list(patient_list[midpoint:])
            
            return self.merge_patient_list(left_sorted, right_sorted)
