import sys
import os
import pandas as pd

def extract_excel_file(file_path):
    try:
        # Read the Excel file into a DataFrame
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_excel_file(excel_file, output_file):
    # Extract the data from the Excel file
    data = extract_excel_file(excel_file)
    if data is None:
        return

    # Initialize the list to store the bursts
    bursts = []

    # Extract 'Start' and 'End' columns and convert them to float
    for index, row in data.iterrows():
        try:
            start = float(str(row['Start']).replace(',', '.'))  # Convert 'Start' to float
            stop = float(str(row['End']).replace(',', '.'))    # Convert 'End' to float
            bursts.append([start, stop])
        except Exception as e:
            print(f"Error processing row {index} in {excel_file}: {e}")

    # Format the output data
    output_data = [os.path.basename(excel_file), bursts]

    # Write the burst data to the output text file
    try:
        with open(output_file, 'w') as f:
            f.write(f"{output_data[0]}\n")
            for burst in bursts:
                f.write(f"{burst[0]}, {burst[1]}\n")
        print(f"Processed {excel_file}, data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

if __name__ == "__main__":
    # Get the Excel file and output file paths from command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python ExtractXlsx.py <excel_file> <output_file>")
        sys.exit(1)

    excel_file = sys.argv[1]
    output_file = sys.argv[2]

    # Call the function to process the Excel file
    process_excel_file(excel_file, output_file)
