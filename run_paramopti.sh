#!/bin/bash

# Activate the conda environment (if using conda)
# conda activate detection_dep

# Run detect_NBs.py
echo "Running detect_NBs.py..."
python /Websites/PaperCode1/detect_NBs.py

# Check if detect_NBs.py ran successfully
if [ $? -eq 0 ]; then
    echo "detect_NBs.py completed successfully."
else
    echo "detect_NBs.py encountered an error."
    exit 1
fi

# Run calc_performance.py
echo "Running calc_performance.py..."
python /Websites/PaperCode1/calc_performance.py

# Check if calc_performance.py ran successfully
if [ $? -eq 0 ]; then
    echo "calc_performance.py completed successfully."
else
    echo "calc_performance.py encountered an error."
    exit 1
fi

echo "All scripts completed successfully."