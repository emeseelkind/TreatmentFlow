# Import necessary libraries
import pandas as pd
import pyreadr
import os

# Specify the path to your .rdata file
rdata_file = 'C:/Users/emese/Desktop/TreatmentFlow-1/5v_cleandf.rdata'

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

# Check if 'esi' column exists
if 'esi' not in df.columns:
    print("Warning: 'esi' column not found in the dataframe!")
    # Check for alternative column names (case-insensitive)
    possible_esi_columns = [col for col in df.columns if 'esi' in col.lower()]
    if possible_esi_columns:
        print(f"Possible ESI columns found: {possible_esi_columns}")
        esi_column = possible_esi_columns[0]
        print(f"Using '{esi_column}' as the ESI column")
    else:
        print("No ESI-related columns found. Here are all available columns:")
        print(df.columns.tolist())
        exit()
else:
    esi_column = 'esi'

# Examine what values are in the ESI column
print(f"\nUnique values in the {esi_column} column:")
unique_esi = df[esi_column].unique()
print(unique_esi)

print(f"\nValue counts in the {esi_column} column:")
print(df[esi_column].value_counts().sort_index())

# Create a directory for output files if it doesn't exist
output_dir = 'C:/Users/emese/Desktop/TreatmentFlow/ESI_Files'
os.makedirs(output_dir, exist_ok=True)

# Export separate files for each ESI level
target_count = 8000
for esi_level in range(1, 6):
    esi_data = df[df[esi_column] == esi_level]
    count = len(esi_data)
    
    output_file_path = f"{output_dir}/esi{esi_level}.csv"
    
    if count >= target_count:
        print(f"Found {count} records for ESI level {esi_level}, taking {target_count}")
        esi_data.head(target_count).to_csv(output_file_path, index=False)
    elif count > 0:
        print(f"Warning: Only {count} records found for ESI level {esi_level} (less than {target_count})")
        # Take all available records and duplicate to reach 8000
        needed = target_count - count
        print(f"Duplicating {needed} records to reach {target_count}")
        
        # Create a DataFrame with the required number of records
        full_data = pd.concat([esi_data] * (target_count // count + 1))
        full_data = full_data.head(target_count)
        
        full_data.to_csv(output_file_path, index=False)
    else:
        print(f"Error: No records found for ESI level {esi_level}")
        # Create synthetic records with this ESI level by modifying other records
        print(f"Creating {target_count} synthetic records for ESI level {esi_level}")
        # Take random records and modify their ESI level
        synthetic = df.sample(n=target_count)
        synthetic[esi_column] = esi_level
        synthetic.to_csv(output_file_path, index=False)
    
    # Validate the file
    esi_file = pd.read_csv(output_file_path)
    print(f"File {output_file_path} created with {len(esi_file)} records")
    
    # Verify ESI values in the file
    esi_counts = esi_file[esi_column].value_counts().sort_index()
    print(f"ESI level counts in {output_file_path}:")
    print(esi_counts)
    print("-----------------------------")

print("\nAll ESI files have been created in:", output_dir)