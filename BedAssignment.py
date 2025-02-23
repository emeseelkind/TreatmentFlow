"""
TreatmentFlow
Constraint Optimization: Hospital Bed Assignment

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

import random
from ortools.init.python import init
from ortools.linear_solver import pywraplp
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.mathopt import LinearConstraint



# declare solver
solver = pywraplp.Solver.CreateSolver("GLOP")
if not solver:
    raise Exception("Could not create solver GLOP")

model = mathopt.Model(name="BedsModel")
    

# variables
"""
The variables in the problem are represented by an assignment matrix H
On the matrix, we are given a total of B beds (columns), and P patients (rows)
Each variable will be a space in this matrix, denoted by H_p_b, and will be a boolean
    This boolean value will show whether the patient p has been assigned the bed b
"""

P = 10 # number of patients
B = 5 # number of beds
H = [[0 for b in range(B)] for p in range(P)] # patient assignment matrix

patients = [random.randint(1,5) for p in range(P)]


# create variables based on assignment matrix
for p in range(P):
    for b in range(B):
        # H[p][b] = solver.IntVar(0, 1, f"H_{p}_{b}") # each matrix position is true or false

        H[p][b] = model.add_binary_variable(name = f"H_{p}_{b}")

# print("Number of vars =", solver.NumVariables())
# print("Variables: ", solver.variables())


# constraints
"""
The constraints in this problem are that each row p in the matrix H must equal 1
    This constraint means that every patient must be assigned (only) one bed
This allows us to allocate patients to beds, knowing that some may be shared
    The multiple assignment of each bed shows which patients will use it
    In practice, these patients that "share" a bed would occupy it sequentially based on priority
"""

# create constraints, where each row equals 1
# constraints = [0 for p in range(P)]

for p in range(P):
    # each row only equals 1: b1 + b2 + ... + bn = 1

    #constraints[p] = solver.Constraint(1, 1, f"ct_{p}")
    # constraints[p] = LinearConstraint()
    # constraints[p].lower_bound = 1
    # constraints[p].upper_bound = 1

    expr = 0

    for b in range(B):
        # each column (bed option) for the patient has coefficient 1
        #constraints[p].SetCoefficient(H[p][b], 1)

        expr = expr + H[p][b]

        # constraints[p].terms.append(LinearConstraint.Term(variable=H[p][b], coefficient=1))

    # print(f"Patient {p}:", expr)

    model_expr = mathopt.LinearExpression(expr)

    # add row (patient) constraint to model
    model.add_linear_constraint((1 <= model_expr) <= 1)

    # model.add_linear_constraint(constraints[p])
    # model.add
    # model.add_constraint(constraints[p])
        
# print("Number of constraints =", solver.NumConstraints())
# print("Constraints: ", [p.name() for p in solver.constraints()])


# objective function
"""
The objective function of this problem is to minimize the total bed share squared
    We must get the sum of the cost of a column, call it "bed_share"
        The cost of a patient assigned to a bed is the inverse of the patient's priority
            If the patient's priority is 5, their cost is 1
            If the patient's priority is 1, their cost is 5 etc.
    We must get the sum of each (bed_share ^ 2), and minimize
This objective function minimizes sharing resources between patients that need urgent care
    Prioritizes focusing care on the most in-need patients
    Punishes sharing resources between multiple high-need patients
"""

# create objective function
bed_share = 0

for b in range(B):

    expr = 0

    for p in range(P):
        
        patient_cost = patients[p] * H[p][b] # int * model var
        expr = expr + patient_cost # add to expression

    bed_share = bed_share + expr * expr

objective_expr = mathopt.QuadraticExpression(bed_share)
model.minimize(objective_expr)


# invoke solver
params = mathopt.SolveParameters(enable_output=True)

result = mathopt.solve(model, mathopt.SolverType.GSCIP, params=params)
if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
    raise RuntimeError(f"model failed to solve: {result.termination}")

# Print some information from the result.
print("MathOpt solve succeeded")
print("Objective value:", result.objective_value())


print("\n---OPTIMAL SOLUTION---")

results_matrix = [[0 for b in range(B)] for p in range(P)]

for p in range(P):
    for b in range(B):
        if round(result.variable_values()[H[p][b]]) == 1:
            # print(f"Patient {p} in bed {b}")
            results_matrix[p][b] = 1
    print(results_matrix[p])

# for var_val in result.variable_values():
#     print(result.variable_values()[var_val])
