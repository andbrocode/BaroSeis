#!/usr/bin/env python3
"""
Script to get data for multiple time periods using config files.
Iterates over example time periods and config_get_data*.yaml files.
"""

import os
import glob
from pathlib import Path
from src.baroseis import baroseis

# Example time periods from the notebook
TIME_PERIODS = [
    ("2024-03-12 09:00", "2024-03-12 18:00"),
    ("2024-03-15 15:00", "2024-03-15 18:00"),
    ("2024-03-16 12:00", "2024-03-16 15:00"),
    ("2024-03-21 13:00", "2024-03-21 16:00"),
    ("2024-03-24 15:00", "2024-03-24 17:00"),
    ("2024-04-23 02:00", "2024-04-23 05:00"),
    ("2024-08-29 15:00", "2024-08-29 17:00"),
]

# Find all config files
config_dir = Path("./config")
config_files = sorted(glob.glob(str(config_dir / "config_get_data*.yaml")))

print(f"Found {len(config_files)} config files:")
for cf in config_files:
    print(f"  - {cf}")

# Create output directories
os.makedirs("./data", exist_ok=True)

# Iterate over config files and time periods
for config_file in config_files:
    
    print(f"\n{'='*60}")
    print(f"Processing config: {config_file}")
    print(f"{'='*60}")
    
    # Load config from YAML
    config = baroseis.load_from_yaml(config_file)
    
    # Iterate over time periods
    for tbeg, tend in TIME_PERIODS:
        print(f"\nProcessing time period: {tbeg} to {tend}")
        
        try:
            # Initialize baroseis object
            bs = baroseis(conf=config)
            
            # Load data for this time period
            bs.load_data(tbeg=tbeg, tend=tend)
            
            # Create file postfix
            file_postfix = f"{config['baro_seed'].split('.')[1]}_"
            file_postfix += f"{config['seis_seeds'][0].split('.')[1]}_"
            file_postfix += f"{bs.config['tbeg'].strftime('%Y%m%d')}"
            
            # Write waveforms to file
            datapath = "./data/"
            filename = f"{file_postfix}.mseed"
            filepath = os.path.join(datapath, filename)
            
            bs.st.write(filepath, format="MSEED")
            print(f"  Saved waveforms to: {filepath}")
            
            # Write config to file
            config_out = bs.config.copy()
            
            # Convert tbeg and tend to strings
            config_out['tbeg'] = str(config_out['tbeg'])
            config_out['tend'] = str(config_out['tend'])
            config_out['t1'] = str(config_out['t1'])
            config_out['t2'] = str(config_out['t2'])
            
            baroseis.store_as_yaml(config_out, f"./config/config_{file_postfix}_sds.yaml")
            print(f"  Saved config to: ./config/config_{file_postfix}_sds.yaml")
            
            # Modify configuration for loading from file
            config_file_out = config_out.copy()
            config_file_out['data_source'] = 'file'
            config_file_out['path_to_baro_data'] = f"./data/{file_postfix}.mseed"
            config_file_out['path_to_seis_data'] = f"./data/{file_postfix}.mseed"
            
            # Update parameters for loading from file
            config_file_out['metadata_correction'] = False
            config_file_out['remove_baro_response'] = False
            config_file_out['remove_seis_sensitivity'] = False
            config_file_out['remove_seis_response'] = False
            
            baroseis.store_as_yaml(config_file_out, f"./config/config_{file_postfix}_file.yaml")
            print(f"  Saved file config to: ./config/config_{file_postfix}_file.yaml")
            
        except Exception as e:
            print(f"  ERROR: Failed to process {tbeg} to {tend}: {str(e)}")
            continue

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")
