"""
TreatmentFlow
Constraint Optimization: Hospital Bed Assignment - MIP Approach

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

import time
from HospitalClasses import HospitalRecords
from HospitalClasses import print_time
from GreedyBedAssignment import Scheduler
from ortools.sat.python import cp_model

# B = 10  # CHANGE VALUE TO CHANGE NUMBER OF BEDS
# P = 75  # CHANGE VALUE TO CHANGE NUMBER OF PATIENTS

# user inputs for custom model showcasing
print()
B = int(input("How many beds        (int): "))
P = int(input("How many patients    (int): "))
time_limit = int(input("Time limit for MIP   (int): "))

hospital = HospitalRecords(B)
hospital.gen_patient_list(P)


"""
Now that problem formulation is complete, we can calculate constraints and solutions given an objective function

Proposed objective function: 
    - minimize the wait times for each patient, making where high-priority patients wait times are minimized the most
        - medium-priority wait times are minimized second, etc.

Proposed constraints:
    - cannot assign a patient to an occupied bed
    - cannot assign a lower priority patient when a higher priority patient is waiting
    - must assign (or queue) every patient in the list of patients
"""

# MIP modelling time tracker
mip_start = time.time()

# create solver
cpmod = cp_model.CpModel()

# create variables
"""
For the variables, we are adding a 'bed' for each minute of the day
-   in each bed (at that minute), the domain is each integer in range(number of patients)
-   if a bed is given the value k, it is occupied by patient k at that minute
"""

print("Adding variables")

timeline = [[0 for b in range(B)] for m in range(1440)] # 1440 minutes in 24 hours
pat_vars = []

for minute in range(1440):
    for bed in range(B):
        timeline[minute][bed] = cpmod.NewIntVar(lb=-1, ub=P-1, name=f"bed_{bed}_at_{minute}") # -1 means empty bed, 0 to P-1 is a patient id
        pat_vars.append(timeline[minute][bed])


# bed assignment constraints
"""
Patient arrival time consideration
-   a patient can only be assigned a bed when they have already arrived at the hospital

Patient to bed exclusivity
-   a patient can only be assigned a bed when there is no other patient in it
-   a patient can only be in one bed at a time (implicity met by contiguity/discharges constraint)

Patient contiguousness
-   a patent being assigned to a bed will be assigned for a contiguous period (must be at most one arrival per patient)

Patient tenure
-   the difference between the final minute where a patient is assigned and the first
    cannot exceed the patient's service time
    -   for every minute where a patient is assigned to a bed (where value is true),
        the variable in the same bed (service_time) minutes ago is not the same patient (or no patient)

-   the difference between the final minute where a patient is assigned and the first
    MUST equal the patient's service time, unless:
    -   (final minute of day - first minute of patient assignment) < service time
"""

# resources for objective function
wait_times = []     # track wait times (weighted) of each patient
penalties = []        # penalize not assigning a patient

# patient-based constraints
for p in range(P):

    # constraint building progress update
    if p < P - 1:
        print(f"Adding constraints for patient {p+1}/{P}", end ="\r")
    else:
        print(f"Adding constraints for patient {p+1}/{P}")


    # building objective function
    wait_times.append(0)
    penalties.append(0)

    # tracking patient information for constraint and objective building
    priority = hospital.patient_list[p].priority
    arrival = hospital.patient_list[p].arrival_time
    service = hospital.patient_list[p].service_time

    arrivals_list = []

    for bed in range(B):
        for minute in range(1440):

            # arrival constraint
            if minute < arrival:

                # the patient cannot be assigned if they have not yet arrived
                cpmod.AddForbiddenAssignments([timeline[minute][bed]], [[p]])

            else: # other constraints are irrelevant if this minute cannot be assigned

                # make and enforce boolean variable for when current minute is patient p
                this_is_p = cpmod.NewBoolVar(f"bed_{bed}_this_is_{p}_at_{minute}")
                cpmod.Add(timeline[minute][bed] == p).OnlyEnforceIf(this_is_p)
                cpmod.Add(timeline[minute][bed] != p).OnlyEnforceIf(this_is_p.Not())

                # first appearance of p, used for sufficient tenure constraint and objective function
                this_first_p = cpmod.NewBoolVar(f"bed_{bed}_this_first_{p}_at_{minute}")

                if minute > 0:

                    # make and enforce boolean variable for when previous minute is patient p
                    last_was_p = cpmod.NewBoolVar(f"bed_{bed}_last_was_{p}_at_{minute}")
                    cpmod.Add(timeline[minute - 1][bed] == p).OnlyEnforceIf(last_was_p)
                    cpmod.Add(timeline[minute - 1][bed] != p).OnlyEnforceIf(last_was_p.Not())

                    # enfore this_first_p when not minute == 0
                    cpmod.AddBoolAnd([last_was_p.Not(), this_is_p]).OnlyEnforceIf(this_first_p)
                    cpmod.AddBoolOr([last_was_p, this_is_p.Not()]).OnlyEnforceIf(this_first_p.Not())

                else:

                    # enforce this_first_p when minute == 0
                    cpmod.Add(this_is_p == 1).OnlyEnforceIf(this_first_p)
                    cpmod.Add(this_is_p == 0).OnlyEnforceIf(this_first_p.Not())

                # support contiguity constraints
                arrivals_list.append(this_first_p)


                # tenure constraints
                if arrival + service <= minute:

                    # the patient cannot occupy a bed for longer than their service time
                    cpmod.AddForbiddenAssignments([timeline[minute][bed], timeline[minute - service][bed]], [(p, p)])

                # set final minute of service to same patient
                if minute + service < 1440:
                    cpmod.Add(timeline[minute + service - 1][bed] == p).OnlyEnforceIf(this_first_p)
                else:
                    cpmod.Add(timeline[1439][bed] == p).OnlyEnforceIf(this_first_p)

                
                # calculate the weighted wait time (for objective function)
                wait_times[p] = wait_times[p] + this_first_p * priority * (minute - arrival) # will be zero if not this_first_p

    # enforcement of contiguity of patient p
    arrivals = cpmod.NewIntVar(0, 1439, f"arrivals_{p}") # could potentially be an arrival at every minute
    
    # enforce arrivals variable: can only have one arrival per patient
    cpmod.Add(arrivals == sum(arrivals_list))
    cpmod.Add(arrivals <= 1)


    # help build objective function

    # add and enforce penalty for when patient goes unassigned
    penalty = cpmod.NewBoolVar(f"penalty_{p}")
    cpmod.Add(arrivals == 0).OnlyEnforceIf(penalty)
    cpmod.Add(arrivals == 1).OnlyEnforceIf(penalty.Not())

    penalties[p] = penalties[p] + penalty * priority * 1440 # will be zero if not penalty


# objective function
"""
Objective function
-   minimize the wait times of patients, with higher-priority patients scaled such 
    that their wait times are more punishing if long (multiply wait times by patient cost)

Practical implementation:
-   for the first appearance of a patient, multiply the difference between this minute
    and their arrival time by their priority level
-   get the sum of these differences and minimize
"""

print("Adding objective function")

objective = sum(wait_times) + sum(penalties)
cpmod.minimize(objective)

# ---------------------------------

# solving to achieve results
# time_limit = 120
print(f"Model complete. Solving now for {time_limit} seconds or less.")

solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
solver.parameters.max_time_in_seconds = time_limit
solver_start = time.time()
status = solver.solve(cpmod)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for minute in range(1440):
        if minute < 600:
            message = f"{print_time(minute)}  - " # space lines evenly for single digit
        else:
            message = f"{print_time(minute)} - "

        for bed in range(B):
            this_value = solver.value(timeline[minute][bed])
            if this_value >= 10: 
                message += f"[{solver.value(timeline[minute][bed])}]"
            elif this_value >= 0:
                message += f"[ {solver.value(timeline[minute][bed])}]" # space lines evenly for single digit
            else:
                message += "[  ]"
        print(message)
else:
    print("No solution found.")

if status == cp_model.OPTIMAL:
    print(f"\nThe above solution is optimal.")
elif status == cp_model.FEASIBLE:
    print(f"\nThe above solution is feasible, but was not proven optimal.")


# solution comparison
print("\n\n---SOLUTION COMPARISONS---")
print("These solutions use the same randomly generated hospital and patients")
print(f"Patients: {P}, Beds: {B}")

# observing MIP solution
print("\n-MIP Solution-")
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Objective value: {int(solver.ObjectiveValue())}")
else:
    print("Could not find a solution using MIP modelling")
print(f"Time of execution: {time.time() - mip_start} ({solver_start - mip_start} building, {time.time() - solver_start} solving)\n")

# message = ""
# for patient in hospital.patient_list:
#     message += f"{patient.id}:{patient.arrival_time}:{patient.service_time} {patient.priority},   "

# print("Patient:arrival:service and priority")
# print(message)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for p in range(P):
        if solver.Value(wait_times[p]):
            # print(f"Patient {p} wait time: {solver.Value(wait_times[p])}.   Optimality: {solver.Value(assignments[p])} - {solver.Value(wait_times[p])} = {solver.Value(assignments[p]) - solver.Value(wait_times[p])}")
            print(f"Patient {p} ({hospital.patient_list[p].priority}) arrival: {print_time(hospital.patient_list[p].arrival_time)}. Wait time: {print_time(int(solver.Value(wait_times[p]) / hospital.patient_list[p].priority))}")
        if solver.Value(penalties[p]):
            print(f"Patient {p} ({hospital.patient_list[p].priority}) not assigned. Arrival: {print_time(hospital.patient_list[p].arrival_time)}. Penalty: {solver.Value(penalties[p])}")

# comparison to greedy bed assignment
print("\n\n-Greedy Solution-")
greedy_start = time.time()
scheduler = Scheduler(hospital)
scheduler.run_hospital(False)

print(f"Objective value: {scheduler.objective()}")
print(f"Time of execution: {time.time() - greedy_start}\n")

for p in range(P):
    patient = hospital.patient_list[p]
    if patient.get_waiting_time() > 0:
        # print(f"Patient {p} wait time: {solver.Value(wait_times[p])}.   Optimality: {solver.Value(assignments[p])} - {solver.Value(wait_times[p])} = {solver.Value(assignments[p]) - solver.Value(wait_times[p])}")
        print(f"Patient {p} ({patient.priority}) arrival: {print_time(patient.arrival_time)}. Wait time: {print_time(patient.get_waiting_time())}")
    elif patient.get_waiting_time() < 0:
        print(f"Patient {p} ({patient.priority}) not assigned. Arrival: {print_time(patient.arrival_time)}. Penalty: {patient.priority * 1440}")
