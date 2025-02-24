# Import necessary libraries
import pandas as pd
import pyreadr
import os

# Specify the path to your .rdata file
rdata_file = 'path/to/your/5v_cleandf.rdata'  # Change this to your file location
csv_file = 'converted_data.csv'  # Output CSV file name

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

# Save to CSV
df.to_csv(csv_file, index=False)
print(f"DataFrame saved as {csv_file}")
