# main.py
import os
import numpy as np
from time import time as timer
from burst_detection import detect_bursts, order_bursts
from metrics_calculation import calculate_performance_metrics
from plot_functions import plot_rast, plot_omisi, plot_conv_omisi, plot_der_conv_omisi
from data_processing import load_and_process_data

def main():
    start_time_total = timer()
    
    # Initialize parameter values
    min_slope_values = np.linspace(0, 0.0007, num=5)
    division_factor_values = np.linspace(1, 20, num=5)
    smoothness_values = np.linspace(0.001, 3, num=5)

    bursts_by_params = {(min_slope, division_factor, smoothness): [] for min_slope in min_slope_values for division_factor in division_factor_values for smoothness in smoothness_values}
    
    # Load and process data
    csv_data_list, csv_file_paths = load_and_process_data()  # Implement the function to load your data
    
    for file_index, data in enumerate(csv_data_list):
        file_name = csv_file_paths[file_index]
        title = os.path.basename(file_name).split('.')[0]
        
        # Perform burst detection and plotting here
        spikes, elecs, time_vector_filtered, conv_mean_overall_filtered, slopes = detect_bursts_and_plot(data)
        
        # Perform performance calculation
        performance_metrics = calculate_performance_metrics(min_slope_values, division_factor_values, smoothness_values, bursts_by_params)
        
        # Save metrics to output (Excel or CSV)
        save_metrics(performance_metrics)
    
    end_time_total = timer()
    total_duration = int(end_time_total - start_time_total)
    print(f"This analysis took {total_duration // 3600} hours, {(total_duration % 3600) // 60} minutes, and {total_duration % 60} seconds.")

if __name__ == "__main__":
    main()
