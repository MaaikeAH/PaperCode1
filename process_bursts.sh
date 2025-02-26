#!/bin/bash

# Define the expert folders (which contain annotated bursts)
EXPERTS=("Output_marloes" "Output_monica" "Output_nina")

# Define the root directory containing the Excel and CSV files
ROOT_DIR="./Output_final"
CSV_ROOT_DIR="./Input/raster"  # Root directory containing CSV files with spike timings

# Define a temporary directory to store the burst data
TEMP_CALC_DIR="temp_calculations"
mkdir -p "$TEMP_CALC_DIR"

# Process the annotated network bursts (Excel files)
for EXPERT in "${EXPERTS[@]}"; do
    echo "Processing annotated network bursts from $EXPERT..."

    # Find all Excel files (*.xlsx or *.xls) in the expert folder and subfolders
    find "$ROOT_DIR/$EXPERT" -type f \( -iname "*.xlsx" -o -iname "*.xls" \) | while read EXCEL_FILE; do
        # Extract file name without the extension (for unique naming in temp folder)
        FILENAME=$(basename "$EXCEL_FILE" .xlsx)
        FILENAME=$(basename "$FILENAME" .xls)

        # Call the Python script to process the Excel file and save the data
        python3 ExtractXlsx.py "$EXCEL_FILE" "$TEMP_CALC_DIR/$FILENAME"_expert_burst_data.txt
    done
done

# Process CSV files for spike timings (call the Python script for CSV processing)
echo "Processing CSV files for spike timings..."

# Call the Python script to process the CSV files
python3 ExtractCSVs.py "$CSV_ROOT_DIR" "$TEMP_CALC_DIR"
