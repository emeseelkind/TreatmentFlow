"""
TreatmentFlow
Constraint Optimization: Hospital Bed Assignment

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence
"""

# variables
"""
The variables in the problem are represented by an assignment matrix H
On the matrix, we are given a total of B beds (columns), and P patients (rows)
Each variable will be a space in this matrix, denoted by H_b_p, and will be a boolean
    This boolean value will show whether the patient p has been assigned the bed b
"""

# constraints
"""
The constraints in this problem are that each row p in the matrix H must equal 1
    This constraint means that every patient must be assigned (only) one bed
This allows us to allocate patients to beds, knowing that some may be shared
    The multiple assignment of each bed shows which patients will use it
    In practice, these patients that "share" a bed would occupy it sequentially based on priority
"""

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