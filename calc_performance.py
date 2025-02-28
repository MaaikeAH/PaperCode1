# Description: This script calculates the performance of the burst detection algorithm for each combination of parameters.

# Import necessary libraries
import numpy as np
import pandas as pd
from itertools import product
import os
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sampling_frequency = 10000
transient = 5
simtime = 125 - transient
epsilon = 1e-6

min_slope_values = np.linspace(0, 0.0007, num=5)
division_factor_values = np.linspace(1, 20, num=5)
smoothness_values = np.linspace(0.001, 3, num=5)

# Load ground truth annotations
def load_annotations(folder):
    annotations = {}
    for file_name in os.listdir(folder):
        if file_name.startswith('burst_timings_') and file_name.endswith('.npy'):
            title = file_name.split('_')[2].split('.')[0]
            annotations[title] = np.load(os.path.join(folder, file_name), allow_pickle=True)
    return annotations

gt_bursts_nina = load_annotations('./Expert_annotations/Output_nina')
gt_bursts_marloes = load_annotations('./Expert_annotations/Output_marloes')
gt_bursts_monica = load_annotations('./Expert_annotations/Output_monica')

# Initialize matrices to store performance metrics
normalized_overlap_matrix = np.zeros((len(min_slope_values), len(division_factor_values), len(smoothness_values)))
sensitivity_matrix = np.zeros((len(min_slope_values), len(division_factor_values), len(smoothness_values)))
specificity_matrix = np.zeros((len(min_slope_values), len(division_factor_values), len(smoothness_values)))

# Define your time vector based on the total duration and sampling rate
total_samples = simtime * sampling_frequency
time = np.linspace(0, simtime, total_samples)

# Directory containing the detected bursts files
detected_bursts_dir = "./temp_files"

# Loop over each detected bursts file
for file_name in tqdm(os.listdir(detected_bursts_dir), desc="Processing files"):
    if file_name.endswith('.npy'):
        title = file_name.split('_')[3].split('.')[0]
        try:
            title = int(title)  # Ensure the title is a valid integer
        except ValueError:
            logging.error(f"Skipping file with invalid title: {file_name}")
            continue

        detected_bursts = np.load(os.path.join(detected_bursts_dir, file_name), allow_pickle=True).item()

        total_iterations = len(min_slope_values) * len(division_factor_values) * len(smoothness_values)
        current_iteration = 0
        last_update_time = time.time()


        for i, min_slope in enumerate(tqdm(min_slope_values, desc="Min slope values", leave=False)):
            for j, division_factor in enumerate(tqdm(division_factor_values, desc="Division factor values", leave=False)):
                for b, smoothness in enumerate(tqdm(smoothness_values, desc="Smoothness values", leave=False)):
                    current_iteration += 1
                    progress_percentage = (current_iteration / total_iterations) * 100
                    current_time = time.time()
                    if current_time - last_update_time >= 15:  # Update every 15 seconds
                        logging.info(f"Processing {title}: {progress_percentage:.2f}% complete")
                        last_update_time = current_time

                    # Initialize bimodal signals for this combination
                    bimodal_signal_ground_truth_nina = np.zeros(total_samples)
                    bimodal_signal_ground_truth_marloes = np.zeros(total_samples)
                    bimodal_signal_ground_truth_monica = np.zeros(total_samples)
                    bimodal_signal_detected = np.zeros(total_samples)

                    # Process the ground truth signal nina
                    if str(title) in gt_bursts_nina:
                        for burst in gt_bursts_nina[str(title)]:
                            start, stop = burst[1], burst[2]
                            start_index = int(start * sampling_frequency)
                            stop_index = int(stop * sampling_frequency)
                            bimodal_signal_ground_truth_nina[start_index:stop_index] = 1

                    # Process the ground truth signal marloes
                    if str(title) in gt_bursts_marloes:
                        for burst in gt_bursts_marloes[str(title)]:
                            start, stop = burst[1], burst[2]
                            start_index = int(start * sampling_frequency)
                            stop_index = int(stop * sampling_frequency)
                            bimodal_signal_ground_truth_marloes[start_index:stop_index] = 1

                    # Process the ground truth signal monica
                    if str(title) in gt_bursts_monica:
                        for burst in gt_bursts_monica[str(title)]:
                            start, stop = burst[1], burst[2]
                            start_index = int(start * sampling_frequency)
                            stop_index = int(stop * sampling_frequency)
                            bimodal_signal_ground_truth_monica[start_index:stop_index] = 1

                    # Process detected bursts for this combination of min_slope, division_factor, and smoothness
                    for start, stop in detected_bursts[(min_slope, division_factor, smoothness)]:
                        start_index = int(start * sampling_frequency)
                        stop_index = int(stop * sampling_frequency)
                        bimodal_signal_detected[start_index:stop_index] = 1

                    # Compute performance metrics for expert 1
                    TP_1 = sum(1 for x, y in zip(bimodal_signal_ground_truth_nina, bimodal_signal_detected) if x == 1 and y == 1)
                    TN_1 = sum(1 for x, y in zip(bimodal_signal_ground_truth_nina, bimodal_signal_detected) if x == 0 and y == 0)
                    FP_1 = sum(1 for x, y in zip(bimodal_signal_ground_truth_nina, bimodal_signal_detected) if x == 0 and y == 1)
                    FN_1 = sum(1 for x, y in zip(bimodal_signal_ground_truth_nina, bimodal_signal_detected) if x == 1 and y == 0)

                    normalized_overlap_1 = (TP_1 + TN_1) / total_samples
                    sensitivity_1 = TP_1 / (TP_1 + FN_1 + epsilon)
                    specificity_1 = TN_1 / (TN_1 + FP_1 + epsilon)

                    # Compute performance metrics for expert 2
                    TP_2 = sum(1 for x, y in zip(bimodal_signal_ground_truth_marloes, bimodal_signal_detected) if x == 1 and y == 1)
                    TN_2 = sum(1 for x, y in zip(bimodal_signal_ground_truth_marloes, bimodal_signal_detected) if x == 0 and y == 0)
                    FP_2 = sum(1 for x, y in zip(bimodal_signal_ground_truth_marloes, bimodal_signal_detected) if x == 0 and y == 1)
                    FN_2 = sum(1 for x, y in zip(bimodal_signal_ground_truth_marloes, bimodal_signal_detected) if x == 1 and y == 0)

                    normalized_overlap_2 = (TP_2 + TN_2) / total_samples
                    sensitivity_2 = TP_2 / (TP_2 + FN_2 + epsilon)
                    specificity_2 = TN_2 / (TN_2 + FP_2 + epsilon)

                    # Compute performance metrics for expert 3
                    TP_3 = sum(1 for x, y in zip(bimodal_signal_ground_truth_monica, bimodal_signal_detected) if x == 1 and y == 1)
                    TN_3 = sum(1 for x, y in zip(bimodal_signal_ground_truth_monica, bimodal_signal_detected) if x == 0 and y == 0)
                    FP_3 = sum(1 for x, y in zip(bimodal_signal_ground_truth_monica, bimodal_signal_detected) if x == 0 and y == 1)
                    FN_3 = sum(1 for x, y in zip(bimodal_signal_ground_truth_monica, bimodal_signal_detected) if x == 1 and y == 0)

                    normalized_overlap_3 = (TP_3 + TN_3) / total_samples
                    sensitivity_3 = TP_3 / (TP_3 + FN_3 + epsilon)
                    specificity_3 = TN_3 / (TN_3 + FP_3 + epsilon)

                    # Average the performance metrics across all experts
                    normalized_overlap = (normalized_overlap_1 + normalized_overlap_2 + normalized_overlap_3) / 3
                    sensitivity = (sensitivity_1 + sensitivity_2 + sensitivity_3) / 3
                    specificity = (specificity_1 + specificity_2 + specificity_3) / 3

                    # Store in matrices
                    normalized_overlap_matrix[i, j, b] = normalized_overlap
                    sensitivity_matrix[i, j, b] = sensitivity
                    specificity_matrix[i, j, b] = specificity

# Generate parameter combinations (assuming 5 values for each parameter)
param1_values = min_slope_values  # min slope
param2_values = division_factor_values  # division factor
param3_values = smoothness_values  # smoothness

# Generate all combinations of parameter values
parameter_combinations = list(product(param1_values, param2_values, param3_values))

# Flatten the 3D arrays (to get one value per parameter combination)
normalized_overlap_flat = normalized_overlap_matrix.flatten()
sensitivity_flat = sensitivity_matrix.flatten()
specificity_flat = specificity_matrix.flatten()

# Create DataFrame to store the results
df = pd.DataFrame(parameter_combinations, columns=['Min. Slope', 'Division Factor', 'Smoothness'])
df['Normalized Overlap'] = normalized_overlap_flat
df['Sensitivity'] = sensitivity_flat
df['Specificity'] = specificity_flat

# Write to Excel
output_file = 'performance_metrics.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Parameter-Performance Matrix')

print("Data written to Excel.")