#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute pressure-tilt coefficients for ROMY data in time windows.

This script:
1. Loads data for specified time periods
2. Processes data in overlapping windows
3. Computes and stores pressure-tilt coefficients
4. Generates plots for each window

Author: Andreas Brotzer
Date: October 2025
"""

import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime, timedelta
from obspy import UTCDateTime
from pathlib import Path
from multiprocessing import Lock
from tqdm import tqdm
from obspy.signal.cross_correlation import correlate, xcorr_max

from src.baroseis import baroseis
from src.plots.plot_waveforms import plot_waveforms
from src.plots.plot_residuals import plot_residuals

# Global lock for file access
file_lock = Lock()

def load_config(config_file: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert time strings to UTCDateTime objects
    for key in ['tbeg', 'tend']:
        if key in config and config[key] is not None:
            config[key] = UTCDateTime(config[key])
            
    return config

def generate_time_windows(config: dict) -> list:
    """
    Generate overlapping time windows for analysis.
    
    Args:
        config: Configuration dictionary containing:
            - tbeg: Start time
            - tend: End time
            - window_hours: Length of each window in hours
            - window_overlap: Fraction of overlap between windows (0-1)
            - buffer_hours: Additional buffer time in hours for each window
        
    Returns:
        List of tuples containing (window_start, window_end, buffer_start, buffer_end)
    """
    windows = []
    window_seconds = config['window_hours'] * 3600
    step_seconds = window_seconds * (1 - config['window_overlap'])
    buffer_seconds = config['buffer_hours'] * 3600
    
    current_time = config['tbeg']
    while current_time + window_seconds <= config['tend']:
        # Window times without buffer
        win_start = current_time
        win_end = current_time + window_seconds
        
        # Window times with buffer
        buf_start = win_start - buffer_seconds
        buf_end = win_end + buffer_seconds
        
        windows.append((win_start, win_end, buf_start, buf_end))
        current_time += step_seconds
        
    return windows

def compute_cross_correlation(seismic_data: np.ndarray, pressure_data: np.ndarray, max_shift: int = 100) -> dict:
    """
    Compute cross-correlation between seismic component and pressure data using ObsPy.
    
    Args:
        seismic_data: Seismic component data (N, E, or Z)
        pressure_data: Pressure data (P) or Hilbert data (H)
        max_shift: Maximum number of samples to shift during cross-correlation
        
    Returns:
        Dictionary containing cc_zero, cc_max, cc_lag
    """
    # Remove mean to focus on correlation structure
    seis_centered = seismic_data - np.mean(seismic_data)
    press_centered = pressure_data - np.mean(pressure_data)
    
    # Compute cross-correlation using ObsPy
    cc = correlate(seis_centered, press_centered, shift=max_shift)
    
    # Zero-lag cross-correlation (at index max_shift)
    cc_zero = cc[max_shift]
    
    # Maximum cross-correlation and corresponding lag using ObsPy
    cc_lag, cc_max = xcorr_max(cc)
    
    return {
        'cc_zero': cc_zero,
        'cc_max': cc_max,
        'cc_lag': cc_lag
    }

def process_window(args: tuple) -> dict:
    """
    Process a single time window.
    
    Args:
        args: Tuple containing (config, window_times, output_dir, results_file)
        
    Returns:
        Dictionary containing processing results
    """
    config, window_times, output_dir, results_file = args
    win_start, win_end, buf_start, buf_end = window_times
    
    try:
        # Initialize baroseis object
        bs = baroseis(config)
        
        # Load data with buffer
        bs.load_data(tbeg=UTCDateTime(buf_start), tend=UTCDateTime(buf_end))
        
        # Filter and process data
        bs.filter_data(fmin=config['fmin'], fmax=config['fmax'])
        bs.st.detrend('demean')
        bs.st.taper(0.1, 'cosine')
        
        if config['integrate_data']:
            # Integrate rotation rate to tilt
            bs.integrate_data(method="cumtrapz")
        
        if config['type'] == "translation":
            # turn acceleration into tilt wiht g = 9.81 m/s² and -1 due to definition of acceleration
            for tr in bs.st:
                if tr.stats.channel[1] == "H":
                    tr.stats.channel = tr.stats.channel[0] + "A" + tr.stats.channel[-1]
                    if tr.stats.channel[-1] in ["N", "E"]:
                        tr.data = -tr.data/9.81

        # Trim to actual window without buffer
        bs.st = bs.st.trim(win_start, win_end)
        bs.st.detrend('demean')
        bs.st.taper(0.05, 'cosine')
        
        # Make waveform plot
        # if config['save_plots']:
        #     try:
        #         fig = bs.plot_waveforms(time_unit="minutes", channel_type="A", out=True)
        #         with file_lock:
        #             fig.savefig(os.path.join(output_dir, f'waveforms_{win_start.strftime("%Y%m%d_%H%M")}.png'))
        #         plt.close(fig)
        #     except:
        #         print(f"Could not plot waveforms for {win_start.strftime('%Y%m%d_%H%M')}")

        # Predict tilt from pressure
        bs.predict_tilt_from_pressure(method="least_squares", channel_type="A", zero_intercept=True, verbose=False)

        # Plot residuals
        if config['save_plots']:
            try:
                fig = bs.plot_residuals(time_unit="minutes", channel_type="A", out=True)
                with file_lock:
                    fig.savefig(os.path.join(output_dir, f'residuals_{win_start.strftime("%Y%m%d_%H%M")}.png'))
                plt.close(fig)
            except:
                print(f"Could not plot residuals for {win_start.strftime('%Y%m%d_%H%M')}")

        # Prepare results
        results = {
            'window_start': win_start.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'window_end': win_end.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            # Model results
            'P_coef_N': np.nan,
            'H_coef_N': np.nan,
            'var_red_N': np.nan,
            'P_coef_E': np.nan,
            'H_coef_E': np.nan,
            'var_red_E': np.nan,
            'P_coef_Z': np.nan,
            'H_coef_Z': np.nan,
            'var_red_Z': np.nan,
            # Cross-correlation results
            'cc_zero_N_P': np.nan,
            'cc_max_N_P': np.nan,
            'cc_lag_N_P': np.nan,
            'cc_zero_N_H': np.nan,
            'cc_max_N_H': np.nan,
            'cc_lag_N_H': np.nan,
            'cc_zero_E_P': np.nan,
            'cc_max_E_P': np.nan,
            'cc_lag_E_P': np.nan,
            'cc_zero_E_H': np.nan,
            'cc_max_E_H': np.nan,
            'cc_lag_E_H': np.nan,
            'cc_zero_Z_P': np.nan,
            'cc_max_Z_P': np.nan,
            'cc_lag_Z_P': np.nan,
            'cc_zero_Z_H': np.nan,
            'cc_max_Z_H': np.nan,
            'cc_lag_Z_H': np.nan,
        }

        # Add coefficients for each component
        try:
            for comp in ['N', 'E', 'Z']:
                results[f'P_coef_{comp}'] = bs.p_coefficient.get(comp, np.nan)
                results[f'H_coef_{comp}'] = bs.h_coefficient.get(comp, np.nan)

                # Get variance reduction
                try:
                    tr_rot = bs.st.select(channel=f'*A{comp}')[0]
                    tr_pred = bs.st.select(location="PP", channel=f'*A{comp}')[0]
                 
                    # Compute variance reduction
                    var_red = bs.variance_reduction(tr_rot.data, tr_rot.data - tr_pred.data)
                    results[f'var_red_{comp}'] = var_red
                except:
                    results[f'var_red_{comp}'] = np.nan
        except:
            print(f"Could not add coefficients for {win_start.strftime('%Y%m%d_%H%M')}")

        # Compute cross-correlations between seismic components and P/H
        try:
            # Get pressure data (P)
            pressure_data = bs.st.select(channel="*DO")[0].data
            
            # Get Hilbert data (H) - if available
            try:
                hilbert_data = bs.st.select(channel="*DH")[0].data
            except:
                hilbert_data = None
            
            # Compute cross-correlations for each seismic component
            for comp in ['N', 'E', 'Z']:
                try:
                    # Get seismic component data
                    seis_data = bs.st.select(channel=f'*A{comp}')[0].data
                    
                    # Cross-correlation with pressure (P)
                    cc_p = compute_cross_correlation(seis_data, pressure_data)
                    results[f'cc_zero_{comp}_P'] = cc_p['cc_zero']
                    results[f'cc_max_{comp}_P'] = cc_p['cc_max']
                    results[f'cc_lag_{comp}_P'] = cc_p['cc_lag']
                    
                    # Cross-correlation with Hilbert (H) if available
                    if hilbert_data is not None:
                        cc_h = compute_cross_correlation(seis_data, hilbert_data)
                        results[f'cc_zero_{comp}_H'] = cc_h['cc_zero']
                        results[f'cc_max_{comp}_H'] = cc_h['cc_max']
                        results[f'cc_lag_{comp}_H'] = cc_h['cc_lag']
                        
                except Exception as e:
                    print(f"Could not compute cross-correlation for component {comp}: {str(e)}")
                    
        except Exception as e:
            print(f"Could not compute cross-correlations for {win_start.strftime('%Y%m%d_%H%M')}: {str(e)}")

        # Save results to CSV with lock
        df = pd.DataFrame([results])
        with file_lock:
            if os.path.exists(results_file):
                df.to_csv(results_file, mode='a', header=False, index=False)
            else:
                df.to_csv(results_file, index=False)

        return results

    except Exception as e:
        print(f"Error processing window {win_start} - {win_end}: {str(e)}")
        return None

def main():
    """Main function to run the coefficient computation workflow."""
    
    # Load configuration
    config_file = sys.argv[1]
    # config_file = "config_compute_model_coefficients.yaml"
    config = load_config(config_file)
    
    # Create output directory
    output_dir = Path(config.get('output_dir', 'output/coefficients'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results file path
    results_file = output_dir / 'coefficients.csv'
    
    # Generate time windows
    windows = generate_time_windows(config)
    
    # Determine number of processes
    n_cores = config.get('n_cores', 0)
    if n_cores <= 0:
        n_cores = mp.cpu_count()
    
    # Prepare arguments for parallel processing
    process_args = [(config, window, str(output_dir), str(results_file)) for window in windows]
    
    # Process windows in parallel with progress bar
    print(f"\nProcessing {len(windows)} windows using {n_cores} cores...")
    time.sleep(1)
    start_time = time.time()

    with mp.Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_window, process_args),
            total=len(windows),
            desc="Processing windows"
        ))

    print(f"Processing time: {(time.time() - start_time)/60} minutes")

    # Count successful windows
    successful = sum(1 for r in results if r is not None)
    print(f"\nProcessing complete: {successful}/{len(windows)} windows processed successfully")
    print(f"Results saved in {output_dir}")

if __name__ == '__main__':
    # Required for Windows compatibility
    mp.freeze_support()
    main()