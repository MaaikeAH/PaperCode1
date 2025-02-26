import os
import pandas as pd
import sys

def extract_csv_file(filename):
    # Load the CSV data and assume it's in a single column or raw data
    data = pd.read_csv(filename, header=None)  # No header since there are no columns
    return data.values  # Return the data as is, without flattening

def process_csv_files(root_directory, output_dir):
    # Create a list to store the extracted spike timings from each CSV
    all_spike_timings = []
    
    # Use os.walk to iterate through all subdirectories and find CSV files
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file).replace('\\', '/')
                spike_timings = extract_csv_file(file_path)
                all_spike_timings.append(spike_timings)

    # Save the spike timings to a file in the output directory
    output_file = os.path.join(output_dir, 'extracted_spike_timings.txt')
    with open(output_file, 'w') as f:
        for timings in all_spike_timings:
            for timing in timings:
                f.write(f"{timing}\n")  # Save each spike timing on a new line
    print(f"Spike timings saved to {output_file}")

if __name__ == "__main__":
    # Check if we are getting the root directory as an argument
    if len(sys.argv) != 3:
        print("Usage: python ExtractCSVs.py <csv_root_directory> <output_directory>")
        sys.exit(1)

    # Extract the root directory for CSV files and the output directory
    csv_root_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    # Process the CSV files and extract the spike timings
    process_csv_files(csv_root_directory, output_directory)
