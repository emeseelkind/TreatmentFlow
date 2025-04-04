import pandas as pd
import pyreadr
import os

# Specify the path to your .rdata file
rdata_file = 'C:/Users/emese/Desktop/TreatmentFlow-1/5v_cleandf.rdata'
output_folder = 'C:/Users/emese/Desktop/TreatmentFlow/CTAS_files'

# Check if the file exists
if not os.path.exists(rdata_file):
    print(f"File not found: {rdata_file}")
    exit()

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

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

# Generate five different hospital data files with different starting indices
for i in range(10):
    sampled_df = df.iloc[i::75]  # Select every 75th row starting at index i
    output_file_path = os.path.join(output_folder, f'hospital_data_{i+1}.csv')
    sampled_df.to_csv(output_file_path, index=False)
    print(f"DataFrame saved as {output_file_path}")
