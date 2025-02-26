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

# define function for plotting the raster plot
def plot_rast(spikes,elecs,title):
    # formatting raster plot
    plt.figure(figsize=(10,3))
    plt.title(title)
    plt.yticks(np.arange(min(elecs), max(elecs)+1, 1.0))
    plt.xlabel("Time (s)")
    plt.ylabel("Electrode number")

    # plotting raster plot
    plt.xlim(0, 120) 
    plt.scatter(x = spikes, y = elecs, marker = "|", color = 'grey', alpha = 0.5)
    plt.tight_layout(pad=2.0)  # Adjust the padding of the figure
#     plt.savefig(os.path.join(directory_path, f"{title}_raster_plot.jpg"), dpi=300)
    plt.show()
    
        
# Plot overall mean ISI
def plot_omisi(time_vector_filtered,mean_overall_filtered,mean_oisi):
    plt.figure(figsize = (10,3))
    plt.plot(time_vector_filtered,mean_overall_filtered,color = 'grey')
#     plt.axhline(mean_oisi, label = "Threshold", color = 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Overall mean ISI (s)')
    plt.title('Mean ISI plot')
    plt.legend(loc = 2)
    plt.xlim(0,120)
    plt.tight_layout(pad=1.0)  # Adjust the padding of the figure
#     plt.savefig(os.path.join(directory_path, f"{title}_omisi_withoutthr.jpg"), dpi=300)
    plt.show()
    
# Plot convolved overall mean ISI
def plot_conv_omisi(time_vector_filtered,conv_mean_overall_filtered,conv_mean_oisi):
    plt.figure(figsize = (10,3))
    plt.plot(time_vector_filtered,conv_mean_overall_filtered, color = 'grey')
    plt.axhline(conv_mean_oisi, label = "Threshold", color = 'r', linestyle = '--')
    plt.xlabel('Time (s)')
    plt.ylabel('Overall mean ISI (s)')
    plt.title('Convolved Mean ISI')
    plt.legend(loc = 2)
    plt.xlim(0,120)
    plt.tight_layout(pad=1.0)  # Adjust the padding of the figure
#     plt.savefig(os.path.join(directory_path, f"{title}_conv_omisi_withoutthr.jpg"), dpi=300)
    plt.show()

# Plot convolved derivative of overall mean
def plot_der_conv_omisi(time_vector_filtered,der_mean_overall):
    plt.figure(figsize = (10,3))
    plt.plot(time_vector_filtered,der_mean_overall, color = 'grey')
    plt.xlabel('Time (s)')
    plt.ylabel('Derivative')
    plt.title('Derivative of convolved Mean ISI')
    plt.axhline(y=0.0001, color = 'blue', linestyle = '--', label = "Minimum Slope Variation")
    plt.axhline(y=-0.0001, color = 'blue', linestyle = '--')
    plt.axhline(y=0, color = 'black')
    plt.xlim(0,120)
    plt.legend(loc = 2)
    plt.tight_layout(pad=1.0)  # Adjust the padding of the figure
#     plt.savefig(os.path.join(directory_path, f"{title}_der_conv_omisi_withthr.jpg"), dpi=300)
    plt.show

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

# Assign burst numbers
def order_bursts(bursts):
    burst_number = 1
    burststarts = bursts[:,0]
    burstends = bursts[:,1]
    ordered = []
    
    for start, end in zip(burststarts, burstends):
        ordered.append((burst_number, start, end))
        burst_number += 1
        
    return np.array(ordered)

# Constants and parameter ranges
epsilon = 1e-10  # Prevent division by zero
sampling_frequency = 10000
transient = 5
simtime = 125 - transient

min_slope_values = np.linspace(0, 0.0007, num=5)
division_factor_values = np.linspace(1, 20, num=5)
smoothness_values = np.linspace(0.001,3, num=5)

bursts_by_params = {(min_slope, division_factor,smoothness): [] for min_slope in min_slope_values for division_factor in division_factor_values for smoothness in smoothness_values}
no_matrices = []
sens_matrices = []
spec_matrices = []

start_time_total = timer()  # Record the start time

# Example loop structure to process files and update the metrics array
for file_index, data in enumerate(csv_data_list):  # Assuming csv_data_list is defined
    start_time1 = timer()  # Record the start time
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
    
    # Plot raster plot
    rasterplot = plot_rast(spikes, elecs, title)

    for i, min_slope in enumerate(min_slope_values):
        for j, division_factor in enumerate(division_factor_values):
            for b,smoothness in enumerate(smoothness_values):
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
                # mean_oisi = np.nanmean(mean_overall_filtered)/division_factor # non-convolved
                conv_mean_oisi = np.nanmean(conv_mean_overall_filtered)/division_factor # convolved
    
                # Plot mean_overall_filtered (without convolving)
            #     print(mean_oisi)
                # omisi_plot = plot_omisi(time_vector_filtered,mean_overall_filtered,mean_oisi)
    
            #     Plot convolved mean_overall_filtered
            #     print(conv_mean_oisi)
                # c_omisi_plot = plot_conv_omisi(time_vector_filtered,conv_mean_overall_filtered,conv_mean_oisi)    
    
                # calculate derivative of overall mean ISI(s)
                der_mean_overall = np.gradient(conv_mean_overall_filtered)   
    
                # plot derivative of overall mean ISI(s)
    #             der_plot = plot_der_conv_omisi(time_vector_filtered,der_mean_overall)
    
                # Take absolute values of slopes
                slopes = [abs(l) for l in der_mean_overall]
    
                bursts_mean_isi = detect_bursts(conv_mean_overall_filtered,slopes,time_vector_filtered,min_slope)
    
                bursts_by_params[(min_slope, division_factor)] = bursts_mean_isi

    end_time_method = timer()  # Record the end time
    total_duration_method = int(end_time_method - start_time1)  # Calculate duration in seconds and convert to integer
    
    # Convert seconds to hours, minutes, and seconds
    hours_m = total_duration_method // 3600
    minutes_m = (total_duration_method % 3600) // 60
    seconds_m = total_duration_method % 60
    
    print(f'Running Mean ISI method for 2500 parameter combinations takes:{hours_m} hours, {minutes_m} minutes, and {seconds_m} seconds.')
    
    #performance evaluation
    normalized_overlap_matrix = np.zeros((len(min_slope_values), len(division_factor_values), len(smoothness_values)))
    sensitivity_matrix = np.zeros((len(min_slope_values), len(division_factor_values), len(smoothness_values)))
    specificity_matrix = np.zeros((len(min_slope_values), len(division_factor_values), len(smoothness_values)))
    
    # Define your time vector based on the total duration and sampling rate
    total_samples = simtime * sampling_frequency
    time = np.linspace(0, simtime, total_samples)
    
    # Prepare the figure with subplots for each combination of parameters
    num_combinations = len(min_slope_values) * len(division_factor_values) * len(smoothness_values)
    
    for i,min_slope in enumerate(min_slope_values):
        for j,division_factor in enumerate(division_factor_values):
            for b,smoothness in enumerate(smoothness_values):
                
                # Initialize bimodal signals for this combination
                bimodal_signal_ground_truth_nina = np.zeros(total_samples)
                bimodal_signal_ground_truth_marloes = np.zeros(total_samples)
                bimodal_signal_ground_truth_monica = np.zeros(total_samples)

                bimodal_signal_detected = np.zeros(total_samples)
    
                # Process the ground truth signal nina
                for burst in gt_bursts_nina:
                    if burst[0] == title:
                        for start, stop in burst[1]:
                            start_index = int(start * sampling_frequency)
                            stop_index = int(stop * sampling_frequency)
                            bimodal_signal_ground_truth_nina[start_index:stop_index] = 1

                # Process the ground truth signal marloes
                for burst in gt_bursts_marloes:
                    if burst[0] == title:
                        for start, stop in burst[1]:
                            start_index = int(start * sampling_frequency)
                            stop_index = int(stop * sampling_frequency)
                            bimodal_signal_ground_truth_marloes[start_index:stop_index] = 1

                # Process the ground truth signal monica
                for burst in gt_bursts_monica:
                    if burst[0] == title:
                        for start, stop in burst[1]:
                            start_index = int(start * sampling_frequency)
                            stop_index = int(stop * sampling_frequency)
                            bimodal_signal_ground_truth_monica[start_index:stop_index] = 1
    
                # Process detected bursts for this combination of min_slope, division_factor, and smoothness
                for start, stop in bursts_by_params[(min_slope, division_factor, smoothness)]:
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
            
        no_matrices.append(normalized_overlap_matrix)
        sens_matrices.append(sensitivity_matrix)
        spec_matrices.append(specificity_matrix)
        
        end_time1 = timer()  # Record the end time
        total_duration_seconds = int(end_time1 - start_time1)  # Calculate duration in seconds and convert to integer
        
        # Convert seconds to hours, minutes, and seconds
        hours = total_duration_seconds // 3600
        minutes = (total_duration_seconds % 3600) // 60
        seconds = total_duration_seconds % 60
        
        print(f"Iteration {title} took {hours} hours, {minutes} minutes, and {seconds} seconds.")
    
    end_time_total = timer()
    total_duration = int(end_time_total-start_time_total)
    hours2 = total_duration // 3600
    minutes2 = (total_duration % 3600) // 60
    seconds2 = total_duration % 60
    
    print(f"This analysis took {hours2} hours, {minutes2} minutes, and {seconds2} seconds.")
    
# Getting mean values
no_arrays = [no_matrices[i] for i in range(1, len(no_matrices), 2)]
mean_no_array = np.mean(no_arrays, axis=0)

sens_arrays = [sens_matrices[i] for i in range(1, len(sens_matrices), 2)]
mean_sens_array = np.mean(sens_arrays, axis=0)

spec_arrays = [spec_matrices[i] for i in range(1, len(spec_matrices), 2)]
mean_spec_array = np.mean(spec_arrays, axis=0)

# Generate parameter combinations (assuming 5 values for each parameter)
param1_values = min_slope_values # min slope
param2_values = division_factor_values # division factor
param3_values = smoothness_values #smoothness

# Generate all combinations of parameter values
parameter_combinations = list(product(param1_values, param2_values, param3_values))

# Flatten the 3D arrays (to get one value per parameter combination)
mean_no_flat = mean_no_array.flatten()
mean_sens_flat = mean_sens_array.flatten()
mean_spec_flat = mean_spec_array.flatten()

# Create DataFrame to store the results
df = pd.DataFrame(parameter_combinations, columns=['Min. Slope', 'Division Factor', 'Smoothness'])
df['Mean_NO'] = mean_no_flat
df['Mean_Sensitivity'] = mean_sens_flat
df['Mean_Specificity'] = mean_spec_flat

# Write to Excel
output_file = 'mean_matrix_with_params(5x5x5).xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Parameter-Performance Matrix')

print("Data written to Excel.")