# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import norm
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
from itertools import combinations
import seaborn as sns
from tqdm import tqdm
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Extract .csv files
def extract_csv_file(filename):
    data = pd.read_csv(filename)
    return data

# Mean ISI
# Gausssian smoothing function
def gaussian_kernel(smoothness):
    sigma = smoothness
    size = int(6 * sigma)  # Make size proportional to sigma (e.g., 6*sigma covers ~99% of the Gaussian)
    if size % 2 == 0:
        size += 1  # Ensure size is odd to center the kernel
    x = np.arange(0, size, 1, float)
    x = x - size // 2
    return norm.pdf(x, scale=sigma)

# Detect bursts
def detect_bursts(conv_mean_overall_filtered,slopes,time_vector_filtered,min_slope):
    bursts = []
    burst_start = None
    
    for i in range(1,len(time_vector_filtered)):
            
        if burst_start is None and conv_mean_overall_filtered[i] <= conv_mean_oisi and conv_mean_overall_filtered[i-1] > conv_mean_oisi and slopes[i] >=min_slope:
            burst_start = time_vector_filtered[i]
            
        elif burst_start is not None and conv_mean_overall_filtered[i] > conv_mean_oisi:
            bursts.append([burst_start,time_vector_filtered[i]])
            burst_start = None
    
    return np.array(bursts)

# Constants and parameter ranges
epsilon = 1e-10  # Prevent division by zero
sampling_frequency = 10000
transient = 5
simtime = 125 - transient

min_slope_values = np.linspace(0, 0.0007, num=5)
division_factor_values = np.linspace(1, 20, num=5)
smoothness_values = np.linspace(0.001,3, num=5)

no_matrices = []
sens_matrices = []
spec_matrices = []

# Specify the root directory containing the maps and CSV files
root_directory = "./Input/raster"

# Create the main map to store CSV file paths
csv_file_paths = []
csv_data_list = []

# Use os.walk to iterate through all subdirectories and find CSV files
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file).replace('\\', '/')
            csv_file_paths.append(file_path)
            csv_data = extract_csv_file(file_path)
            csv_data_list.append(csv_data)

# Create a folder for temporary files
temp_folder = "./temp_files"
os.makedirs(temp_folder, exist_ok=True)

# Example loop structure to process files and update the metrics array
for file_index, data in enumerate(tqdm(csv_data_list, desc = "Processing files")):  # Assuming csv_data_list is defined
    try:
        file_name = csv_file_paths[file_index]
        title = os.path.basename(file_name).split('.')[0]
        
        no_matrices.append(title)
        sens_matrices.append(title)
        spec_matrices.append(title)

        # Convert the data to numpy arrays
        data = np.array(data)  # Ensure it's converted to a NumPy array
        data1 = data[data[:, 1] > (transient * sampling_frequency)]  # Remove data before the transient

        spikes = data1[data1[:, 1] > (transient * sampling_frequency), 1].astype(float) / sampling_frequency - transient
        elecs = data1[data1[:, 1] > (transient * sampling_frequency), 0].astype(float)

        # Store the bursts in the dictionary
        bursts_by_params = {}

        total_iterations = len(min_slope_values) * len(division_factor_values) * len(smoothness_values)
        current_iteration = 0
        last_update_time = time.time()

        for i, min_slope in enumerate(min_slope_values):
            for j, division_factor in enumerate(division_factor_values):
                for b,smoothness in enumerate(smoothness_values):
                    current_iteration += 1
                    progress_percentage = (current_iteration / total_iterations) * 100
                    current_time = time.time()
                    if current_time - last_update_time >= 15:  # Update every 15 seconds
                        logging.info(f"Processing {title}: {progress_percentage:.2f}% complete")
                        last_update_time = current_time

                    # Create a time vector spanning the entire duration
                    time_vector = np.arange(0, max(spikes), 1e-4)  # Adjust the time resolution as needed
        
                    # Initialize a list to store ISI arrays for each electrode
                    isi_arrays_list = []
        
                    # Iterate over unique electrode numbers
                    unique_electrodes = np.unique(elecs)
                    for electrode in unique_electrodes:
                        # Extract spike times for the current electrode
                        electrode_spike_times = spikes[elecs == electrode]
        
                        # Calculate ISI for the current electrode
                        isi_array = np.zeros_like(time_vector)  # Initialize ISI array with zeros
        
                        for k in range(len(electrode_spike_times) - 1):
                            spike1 = electrode_spike_times[k]
                            spike2 = electrode_spike_times[k + 1]
                            tisi = spike2 - spike1
        
                            # Find the index in time_vector corresponding to spike1 and spike2
                            idx1 = np.searchsorted(time_vector, spike1)
                            idx2 = np.searchsorted(time_vector, spike2)
        
                            # Fill ISI values in the appropriate range
                            isi_array[idx1:idx2] = tisi
                            if k == 0:
                                isi_array[0:idx1] = np.nan
                            if (k + 1) == (len(electrode_spike_times) - 1):
                                isi_array[idx2:] = np.nan
        
                        # Store the ISI array in the list
                        isi_arrays_list.append(isi_array)
        
                    # Calculate the overall mean ISI signal over time with NaN values
                    mean_overall = np.nanmean(np.array(isi_arrays_list), axis=0)
        
                    # Find the first non-NaN element in the mean_overall array
                    first_non_nan_index = np.argmax(~np.isnan(mean_overall))
        
                    # Extract the time vector and mean ISI values corresponding to non-NaN values
                    time_vector_filtered = time_vector[first_non_nan_index:]
                    mean_overall_filtered = mean_overall[first_non_nan_index:]
        
                    # Convolve the mean ISI signal with the Gaussian kernel
                    kernel = gaussian_kernel(smoothness)
                    kernel /= np.sum(kernel)
                    conv_mean_overall_filtered = np.convolve(mean_overall_filtered, kernel, mode='same')
        
                    # Define thresholds
                    conv_mean_oisi = np.nanmean(conv_mean_overall_filtered)/division_factor # convolved   
        
                    # calculate derivative of overall mean ISI(s)
                    der_mean_overall = np.gradient(conv_mean_overall_filtered)   
        
                    # Take absolute values of slopes
                    slopes = [abs(l) for l in der_mean_overall]
        
                    bursts_mean_isi = detect_bursts(conv_mean_overall_filtered,slopes,time_vector_filtered,min_slope)
        
                    # Store the bursts in the dictionary
                    bursts_by_params[(min_slope, division_factor, smoothness)] = bursts_mean_isi

        # Save the bursts by parameter combinations for the current file
        np.save(os.path.join(temp_folder, f'bursts_by_params_{title}.npy'), bursts_by_params)
        logging.info(f'Successfully processed and saved bursts for {title}')
    except Exception as e:
        logging.error(f'Error processing file {file_name}: {e}')