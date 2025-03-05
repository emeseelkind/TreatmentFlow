"""
TreatmentFlow
Constraint Optimization: Hospital Bed Assignment - MIP Approach

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

from HospitalClasses import HospitalRecords
from ortools.sat.python import cp_model

# running hospital simulation without constraint satisfacation
B = 5  # CHANGE VALUE TO CHANGE NUMBER OF BEDS
P = 5  # CHANGE VALUE TO CHANGE NUMBER OF PATIENTS

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

print("\n\n--- SOLVING SECTION ---")

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
-   a patient can only be in one bed at a time (implicity met by discharges constraint)

Patient contiguousness
-   a patent being assigned to a bed will be assigned for a contiguous period

Patient tenure
-   the difference between the final minute where a patient is assigned and the first
    cannot exceed the patient's service time
    -   for every minute where a patient is assigned to a bed (where value is true),
        the variable in the same bed (service_time) minutes ago is not the same patient (or no patient)

-   the difference between the final minute where a patient is assigned and the first
    MUST equal the patient's service time, unless:
    -   (final minute of day - first minute of patient assignment) < service time
"""

# patient-based constraints
print("Adding constraints")

for p in range(P):
    # print(f"Contiguity of patient {p}")
    arrival = hospital.patient_list[p].arrival_time
    service = hospital.patient_list[p].service_time

    discharges_list = []
    discharges = cpmod.NewIntVar(0, 1439, f"discharges_{p}") # could potentially be a discharge at every minute

    for bed in range(B):
        for minute in range(1440):

            # arrival constraint
            if minute < arrival:

                # the patient cannot be assigned if they have not yet arrived
                cpmod.AddForbiddenAssignments([timeline[minute][bed]], [[p]])

            else: # other constraints are irrelevant if this minute cannot be assigned

                # tenure constraint
                if arrival + service <= minute:

                    # the patient cannot occupy a bed for longer than their service time
                    cpmod.AddForbiddenAssignments([timeline[minute][bed], timeline[minute - service][bed]], [(p, p)])


                # contiguity constraints

                # make and enforce boolean variable for when current minute is patient p
                this_is_p = cpmod.NewBoolVar(f"bed_{bed}_this_is_{p}_at_{minute}")
                cpmod.Add(timeline[minute][bed] == p).OnlyEnforceIf(this_is_p)
                cpmod.Add(timeline[minute][bed] != p).OnlyEnforceIf(this_is_p.Not())

                if minute < 1439: # edge case for next_is_p

                    # make and enforce boolean variable for when next minute is patient p
                    next_is_p = cpmod.NewBoolVar(f"bed_{bed}_next_is_{p}_at_{minute}")
                    cpmod.Add(timeline[minute + 1][bed] == p).OnlyEnforceIf(next_is_p)
                    cpmod.Add(timeline[minute + 1][bed] != p).OnlyEnforceIf(next_is_p.Not())
                    
                    # get discharge variable and enforce: current minute is p and next is not
                    discharge = cpmod.NewBoolVar(f"bed_{bed}_discharge_{p}_at_{minute}")
                    cpmod.AddBoolAnd([this_is_p, next_is_p.Not()]).OnlyEnforceIf(discharge)
                    cpmod.AddBoolOr([this_is_p.Not(), next_is_p]).OnlyEnforceIf(discharge.Not())

                    discharges_list.append(discharge)
                
                else:

                    # the patient being in the bed at the final minute should count as a discharge
                    discharges_list.append(this_is_p)

    # enforcement of contiguity
    # enforce discharges variable: can only have one discharge per patient
    cpmod.Add(discharges == sum(discharges_list))
    cpmod.Add(discharges <= 1)


# objective function
"""
Objective function
-   minimize the wait times of patients, with higher-priority patients scaled such 
    that their wait times are more punishing if long (multiply wait times by patient cost)
"""

print("Adding objective function")

weighted_wait_times = 0
for minute in range(1440):
    for bed in range(B):
        for p in range(P):
            arrival = hospital.patient_list[p].arrival_time
            priority = hospital.patient_list[p].priority

            # add wait time * priority to objective expression
            """
            if ([minute - 1] != patient) and ([minute] == patient):
                weighted_wait_times = weighted_wait_times + (minute - arrival) * priority
            """

# testing objective function
sum_of_patients = 0
for var in pat_vars:
    sum_of_patients += var

cpmod.maximize(sum_of_patients)

# ---------------------------------

# solving to achieve results
print("Model complete. Solving now")

solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
solver.parameters.max_time_in_seconds = 15
status = solver.solve(cpmod)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for minute in range(1440):
        message = f"{minute}: "
        for bed in range(B):
            if solver.value(timeline[minute][bed]) >= 0:
                message += f"[{solver.value(timeline[minute][bed])}]"
            else:
                message += "[  ]"
            # print(f"Bed {solver.value(timeline[minute][bed])}")
    
        print(message)
else:
    print("No solution found.")

if status == cp_model.OPTIMAL:
    print(f"\nThe above solution is optimal.")
elif status == cp_model.FEASIBLE:
    print(f"\nThe above solution is feasible, but was not proven optimal.")

message = ""
for patient in hospital.patient_list:
    message += f"{patient.id}:{patient.arrival_time}:{patient.service_time}, "

print("Patient : arrival : service")
print(message)
