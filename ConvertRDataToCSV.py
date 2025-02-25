# Import necessary libraries
import pandas as pd
import pyreadr
import os

# Specify the path to your .rdata file
rdata_file = 'C:/Users/emese/Desktop/TreatmentFlow-1/5v_cleandf.rdata'
# csv_file = 'hospital_triage_patient_data.csv'  # Output CSV file name

# Check if the file exists
if not os.path.exists(rdata_file):
    print(f"File not found: {rdata_file}")
    exit()

# Load the .rdata file
result = pyreadr.read_r(rdata_file)

# Extract the DataFrame (assuming the R data frame is named 'df')
df = None
for key in result.keys():
    print(f"Loading DataFrame: {key}")
    df = result[key]
    break

# Check if DataFrame was loaded
if df is None:
    print("No DataFrame found in the .rdata file.")
    exit()

# Display the first few rows
print("First 5 rows of the DataFrame:")
print(df.head())

# file is too bit so we extract every 5th row
# Extract every other row (start at index 0 and step by 2)
every_other_row = df.iloc[::75]

# Specify the output file path
output_file_path = 'C:/Users/emese/Desktop/TreatmentFlow/new_hospital_data_75.csv'

# Save the result to a new CSV file
every_other_row.to_csv(output_file_path, index=False)

# Save to CSV
# df.to_csv(csv_file, index=False)
print(f"DataFrame saved as {output_file_path}")
