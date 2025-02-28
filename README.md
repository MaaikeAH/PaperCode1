# Parameter Optimization Script

This repository contains scripts for detecting neural bursts and calculating performance metrics. The main script `run_paramopti.sh` runs the detection and performance calculation scripts sequentially.

## Prerequisites

- Python 3.x
- Conda (optional, for managing environments)
- Required Python packages (listed in `environment.yml`)

## Setup

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Set up the conda environment** (optional):
    If you are using conda, create and activate the environment using the provided [environment.yml](http://_vscodecontentref_/4) file:
    ```sh
    conda env create -f environment.yml
    conda activate detection_dep
    ```

3. **Install required Python packages**:
    If you are not using conda, install the required packages using `pip`:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Make the script executable**:
    ```sh
    chmod +x run_paramopti.sh
    ```

2. **Run the script**:
    ```sh
    ./run_paramopti.sh
    ```

    This script performs the following steps:
    - Activates the conda environment (if using conda)
    - Runs [detect_NBs.py](http://_vscodecontentref_/5) to detect neural bursts
    - Checks if [detect_NBs.py](http://_vscodecontentref_/6) ran successfully
    - Runs [calc_performance.py](http://_vscodecontentref_/7) to calculate performance metrics
    - Checks if [calc_performance.py](http://_vscodecontentref_/8) ran successfully

## Script Details

### [run_paramopti.sh](http://_vscodecontentref_/9)

This bash script runs the detection and performance calculation scripts sequentially.