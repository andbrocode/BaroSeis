#!/usr/bin/python3
"""
Test script to validate wind statistics computation and create comparison plots.

This script:
1. Tests the compute_wind_statistics function
2. Loads original FURT data for validation
3. Creates comprehensive comparison plots

Author: Andreas Brotzer
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from obspy import UTCDateTime, Stream

# Import the main computation function
from compute_wind_statistics import (
    compute_wind_statistics,
    load_furt_stream,
    circular_mean,
    circular_median,
    circular_std
)


def test_wind_statistics(
    coefficients_file: str,
    furt_archive_path: str = '/import/bay200/gif_online/FURT/WETTER/',
    sampling_rate: float = 1.0,
    n_example_windows: int = 5,
    output_plot: str = None
):
    """
    Test function to validate wind statistics computation and create comparison plots.
    
    This function:
    1. Computes wind statistics using compute_wind_statistics
    2. Loads original FURT data
    3. Manually computes statistics for example windows
    4. Creates comprehensive comparison plots
    
    Args:
        coefficients_file: Path to coefficients CSV file
        furt_archive_path: Path to FURT weather station archive
        sampling_rate: Sampling rate for FURT data (Hz)
        n_example_windows: Number of example windows to show in detail
        output_plot: Path to save plot. If None, displays interactively.
    
    Returns:
        tuple: (df_results, fig) - Results DataFrame and matplotlib Figure
    """
    print("="*70)
    print("TESTING WIND STATISTICS COMPUTATION")
    print("="*70)
    
    # Step 1: Compute wind statistics
    print("\n[1/4] Computing wind statistics...")
    df_results = compute_wind_statistics(
        coefficients_file=coefficients_file,
        output_file=None,  # Don't save, we'll use the returned DataFrame
        furt_archive_path=furt_archive_path,
        sampling_rate=sampling_rate
    )
    
    # Convert window times to datetime for plotting
    df_results['window_start_dt'] = pd.to_datetime(df_results['window_start'])
    df_results['window_end_dt'] = pd.to_datetime(df_results['window_end'])
    df_results['window_center'] = df_results['window_start_dt'] + (
        df_results['window_end_dt'] - df_results['window_start_dt']
    ) / 2
    
    # Step 2: Manually compute statistics for example windows (loading data per window)
    print(f"\n[2/4] Manually computing statistics for {n_example_windows} example windows...")
    example_windows = df_results.head(n_example_windows)
    manual_stats = []
    daily_cache = {}  # Cache for daily data
    
    for idx, row in example_windows.iterrows():
        window_start = UTCDateTime(row['window_start'])
        window_end = UTCDateTime(row['window_end'])
        
        try:
            # Determine which days we need for this window
            window_start_date = window_start.date
            window_end_date = window_end.date
            
            # Collect all days needed
            days_needed = []
            current_date = window_start_date
            while current_date <= window_end_date:
                date_key = current_date.strftime('%Y-%m-%d')
                if date_key not in daily_cache:
                    days_needed.append(date_key)
                current_date = UTCDateTime(current_date) + 86400
                current_date = current_date.date
            
            # Load data for missing days
            for date_key in days_needed:
                # Load data for this day (with small buffer)
                day_start = UTCDateTime(date_key) - 3600  # 1 hour buffer before
                day_end = UTCDateTime(date_key) + 86400 + 3600  # 1 hour buffer after
                
                try:
                    st_furt = load_furt_stream(
                        starttime=day_start,
                        endtime=day_end,
                        show_raw=False,
                        sampling_rate=sampling_rate,
                        path_to_archive=furt_archive_path
                    )
                    daily_cache[date_key] = st_furt
                except Exception as e:
                    print(f"Warning: Error loading FURT data for {date_key}: {e}")
                    daily_cache[date_key] = None
            
            # Merge streams for all days in the window
            st_window = Stream()
            current_date = window_start_date
            while current_date <= window_end_date:
                date_key = current_date.strftime('%Y-%m-%d')
                if date_key in daily_cache and daily_cache[date_key] is not None:
                    st_window += daily_cache[date_key]
                current_date = UTCDateTime(current_date) + 86400
                current_date = current_date.date
            
            # Extract wind speed and direction traces
            if len(st_window) == 0:
                print(f"Warning: No wind data available for example window {idx+1}")
                continue
            
            try:
                tr_wind_speed = st_window.select(channel='LAW').merge(method=1)[0]
                tr_wind_dir = st_window.select(channel='LAD').merge(method=1)[0]
            except (IndexError, Exception) as e:
                print(f"Warning: Could not extract wind traces for example window {idx+1}: {e}")
                continue
            
            # Trim to exact window
            tr_ws = tr_wind_speed.copy()
            tr_wd = tr_wind_dir.copy()
            tr_ws.trim(window_start, window_end)
            tr_wd.trim(window_start, window_end)
            
            ws_data = tr_ws.data[~np.isnan(tr_ws.data)]
            wd_data = tr_wd.data[~np.isnan(tr_wd.data)]
            wd_data = wd_data % 360  # Normalize to 0-360
            
            if len(ws_data) > 0 and len(wd_data) > 0:
                manual_stats.append({
                    'window_center': row['window_center'],
                    'ws_mean_manual': np.mean(ws_data),
                    'ws_median_manual': np.median(ws_data),
                    'ws_max_manual': np.max(ws_data),
                    'wd_mean_manual': circular_mean(wd_data, degrees=True),
                    'wd_median_manual': circular_median(wd_data, degrees=True),
                    'wd_std_manual': circular_std(wd_data, degrees=True),
                    'ws_mean_computed': row['wind_speed_mean'],
                    'ws_median_computed': row['wind_speed_median'],
                    'ws_max_computed': row['wind_speed_max'],
                    'wd_mean_computed': row['wind_dir_mean'],
                    'wd_median_computed': row['wind_dir_median'],
                    'wd_std_computed': row['wind_dir_std'],
                })
        except Exception as e:
            print(f"Warning: Error processing example window {idx+1}: {e}")
    
    df_manual = pd.DataFrame(manual_stats)
    
    # Step 4: Create comprehensive plots
    print("\n[4/4] Creating comparison plots...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Colors
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    color3 = 'tab:green'
    
    # Plot 1: Wind speed mean over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_results['window_center'], df_results['wind_speed_mean'], 
             'o-', color=color1, markersize=3, alpha=0.7, label='Computed mean')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Wind Speed Mean (m/s)')
    ax1.set_title('Wind Speed Mean Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Wind speed max over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_results['window_center'], df_results['wind_speed_max'], 
             'o-', color=color2, markersize=3, alpha=0.7, label='Computed max')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Wind Speed Max (m/s)')
    ax2.set_title('Wind Speed Max Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Wind direction mean over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df_results['window_center'], df_results['wind_dir_mean'], 
             'o-', color=color3, markersize=3, alpha=0.7, label='Computed mean')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Wind Direction Mean (degrees)')
    ax3.set_title('Wind Direction Mean Over Time')
    ax3.set_ylim(0, 360)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Scatter - Wind speed mean (computed vs manual)
    ax4 = fig.add_subplot(gs[1, 0])
    if len(df_manual) > 0:
        ax4.scatter(df_manual['ws_mean_manual'], df_manual['ws_mean_computed'], 
                   color=color1, alpha=0.7, s=100, edgecolors='black', linewidth=1)
        # Add 1:1 line
        min_val = min(df_manual['ws_mean_manual'].min(), df_manual['ws_mean_computed'].min())
        max_val = max(df_manual['ws_mean_manual'].max(), df_manual['ws_mean_computed'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        ax4.set_xlabel('Manual Computation (m/s)')
        ax4.set_ylabel('Computed Function (m/s)')
        ax4.set_title('Wind Speed Mean: Computed vs Manual')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        # Calculate correlation
        corr = np.corrcoef(df_manual['ws_mean_manual'], df_manual['ws_mean_computed'])[0, 1]
        ax4.text(0.05, 0.95, f'R = {corr:.4f}', transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Scatter - Wind speed max (computed vs manual)
    ax5 = fig.add_subplot(gs[1, 1])
    if len(df_manual) > 0:
        ax5.scatter(df_manual['ws_max_manual'], df_manual['ws_max_computed'], 
                   color=color2, alpha=0.7, s=100, edgecolors='black', linewidth=1)
        min_val = min(df_manual['ws_max_manual'].min(), df_manual['ws_max_computed'].min())
        max_val = max(df_manual['ws_max_manual'].max(), df_manual['ws_max_computed'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        ax5.set_xlabel('Manual Computation (m/s)')
        ax5.set_ylabel('Computed Function (m/s)')
        ax5.set_title('Wind Speed Max: Computed vs Manual')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        corr = np.corrcoef(df_manual['ws_max_manual'], df_manual['ws_max_computed'])[0, 1]
        ax5.text(0.05, 0.95, f'R = {corr:.4f}', transform=ax5.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 6: Scatter - Wind direction mean (computed vs manual)
    ax6 = fig.add_subplot(gs[1, 2])
    if len(df_manual) > 0:
        ax6.scatter(df_manual['wd_mean_manual'], df_manual['wd_mean_computed'], 
                   color=color3, alpha=0.7, s=100, edgecolors='black', linewidth=1)
        min_val = min(df_manual['wd_mean_manual'].min(), df_manual['wd_mean_computed'].min())
        max_val = max(df_manual['wd_mean_manual'].max(), df_manual['wd_mean_computed'].max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        ax6.set_xlabel('Manual Computation (degrees)')
        ax6.set_ylabel('Computed Function (degrees)')
        ax6.set_title('Wind Direction Mean: Computed vs Manual')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        corr = np.corrcoef(df_manual['wd_mean_manual'], df_manual['wd_mean_computed'])[0, 1]
        ax6.text(0.05, 0.95, f'R = {corr:.4f}', transform=ax6.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 7: Histogram of wind speed mean
    ax7 = fig.add_subplot(gs[2, 0])
    valid_ws_mean = df_results['wind_speed_mean'].dropna()
    if len(valid_ws_mean) > 0:
        ax7.hist(valid_ws_mean, bins=30, color=color1, alpha=0.7, edgecolor='black')
        ax7.axvline(valid_ws_mean.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {valid_ws_mean.mean():.2f} m/s')
        ax7.set_xlabel('Wind Speed Mean (m/s)')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Distribution of Wind Speed Mean')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Histogram of wind speed max
    ax8 = fig.add_subplot(gs[2, 1])
    valid_ws_max = df_results['wind_speed_max'].dropna()
    if len(valid_ws_max) > 0:
        ax8.hist(valid_ws_max, bins=30, color=color2, alpha=0.7, edgecolor='black')
        ax8.axvline(valid_ws_max.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {valid_ws_max.mean():.2f} m/s')
        ax8.set_xlabel('Wind Speed Max (m/s)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Distribution of Wind Speed Max')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Histogram of wind direction mean
    ax9 = fig.add_subplot(gs[2, 2])
    valid_wd_mean = df_results['wind_dir_mean'].dropna()
    if len(valid_wd_mean) > 0:
        ax9.hist(valid_wd_mean, bins=36, color=color3, alpha=0.7, edgecolor='black')
        ax9.axvline(valid_wd_mean.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {valid_wd_mean.mean():.1f}°')
        ax9.set_xlabel('Wind Direction Mean (degrees)')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Distribution of Wind Direction Mean')
        ax9.set_xlim(0, 360)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
    
    # Plot 10: Time series comparison for example window
    ax10 = fig.add_subplot(gs[3, :])
    if len(example_windows) > 0:
        # Select first example window
        example_row = example_windows.iloc[0]
        window_start = UTCDateTime(example_row['window_start'])
        window_end = UTCDateTime(example_row['window_end'])
        
        # Load data for this specific window
        window_start_date = window_start.date
        window_end_date = window_end.date
        
        # Collect all days needed
        days_needed = []
        current_date = window_start_date
        while current_date <= window_end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            days_needed.append(date_key)
            current_date = UTCDateTime(current_date) + 86400
            current_date = current_date.date
        
        # Load and merge data for the window
        st_window = Stream()
        for date_key in days_needed:
            day_start = UTCDateTime(date_key) - 3600
            day_end = UTCDateTime(date_key) + 86400 + 3600
            try:
                st_furt = load_furt_stream(
                    starttime=day_start,
                    endtime=day_end,
                    show_raw=False,
                    sampling_rate=sampling_rate,
                    path_to_archive=furt_archive_path
                )
                st_window += st_furt
            except Exception as e:
                print(f"Warning: Error loading data for plot: {e}")
        
        # Extract traces
        try:
            tr_wind_speed = st_window.select(channel='LAW').merge(method=1)[0]
            tr_wind_dir = st_window.select(channel='LAD').merge(method=1)[0]
            
            # Trim to exact window
            tr_ws_ex = tr_wind_speed.copy()
            tr_wd_ex = tr_wind_dir.copy()
            tr_ws_ex.trim(window_start, window_end)
            tr_wd_ex.trim(window_start, window_end)
            
            # Create time axis
            times = [window_start + i * (1/sampling_rate) for i in range(len(tr_ws_ex.data))]
            times_dt = [t.datetime for t in times]
        except (IndexError, Exception) as e:
            print(f"Warning: Could not extract traces for plot: {e}")
            tr_ws_ex = None
            tr_wd_ex = None
            times_dt = []
        
        if tr_ws_ex is not None and tr_wd_ex is not None and len(times_dt) > 0:
            # Plot wind speed
            ax10_twin = ax10.twinx()
            ax10.plot(times_dt, tr_ws_ex.data, color=color1, alpha=0.6, linewidth=0.5, 
                     label='Wind Speed (m/s)')
            ax10.axhline(example_row['wind_speed_mean'], color=color1, linestyle='--', 
                        linewidth=2, label=f'Mean: {example_row["wind_speed_mean"]:.2f} m/s')
            ax10.axhline(example_row['wind_speed_max'], color=color2, linestyle='--', 
                        linewidth=2, label=f'Max: {example_row["wind_speed_max"]:.2f} m/s')
            ax10.set_ylabel('Wind Speed (m/s)', color=color1)
            ax10.tick_params(axis='y', labelcolor=color1)
            ax10.set_xlabel('Time')
            
            # Plot wind direction
            ax10_twin.plot(times_dt, tr_wd_ex.data % 360, color=color3, alpha=0.6, 
                          linewidth=0.5, label='Wind Direction (degrees)')
            ax10_twin.axhline(example_row['wind_dir_mean'], color=color3, linestyle='--', 
                             linewidth=2, label=f'Mean: {example_row["wind_dir_mean"]:.1f}°')
            ax10_twin.set_ylabel('Wind Direction (degrees)', color=color3)
            ax10_twin.tick_params(axis='y', labelcolor=color3)
            ax10_twin.set_ylim(0, 360)
            
            ax10.set_title(f'Example Window: {example_row["window_start"]} to {example_row["window_end"]}')
            ax10.grid(True, alpha=0.3)
            ax10.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax10.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Combine legends
            lines1, labels1 = ax10.get_legend_handles_labels()
            lines2, labels2 = ax10_twin.get_legend_handles_labels()
            ax10.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax10.text(0.5, 0.5, 'No data available for this window', 
                     ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title(f'Example Window: {example_row["window_start"]} to {example_row["window_end"]}')
    
    plt.suptitle('Wind Statistics Computation Validation', fontsize=16, fontweight='bold', y=0.995)
    
    if output_plot:
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_plot}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total windows processed: {len(df_results)}")
    print(f"Windows with valid wind speed data: {df_results['wind_speed_mean'].notna().sum()}")
    print(f"Windows with valid wind direction data: {df_results['wind_dir_mean'].notna().sum()}")
    
    if len(df_manual) > 0:
        print(f"\nExample windows validation (n={len(df_manual)}):")
        print(f"  Wind speed mean - Mean absolute error: "
              f"{np.mean(np.abs(df_manual['ws_mean_manual'] - df_manual['ws_mean_computed'])):.4f} m/s")
        print(f"  Wind speed max - Mean absolute error: "
              f"{np.mean(np.abs(df_manual['ws_max_manual'] - df_manual['ws_max_computed'])):.4f} m/s")
        print(f"  Wind direction mean - Mean absolute error: "
              f"{np.mean(np.abs(df_manual['wd_mean_manual'] - df_manual['wd_mean_computed'])):.4f}°")
    
    print("="*70)
    
    return df_results, fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test wind statistics computation with validation plots"
    )
    parser.add_argument(
        'coefficients_file',
        type=str,
        help='Path to coefficients CSV file'
    )
    parser.add_argument(
        '--furt-path',
        type=str,
        default='/import/bay200/gif_online/FURT/WETTER/',
        help='Path to FURT weather station archive'
    )
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=1.0,
        help='Sampling rate for FURT data (Hz, default: 1.0)'
    )
    parser.add_argument(
        '--n-examples',
        type=int,
        default=5,
        help='Number of example windows for validation (default: 5)'
    )
    parser.add_argument(
        '-o', '--output-plot',
        type=str,
        default=None,
        help='Path to save plot. If not provided, displays interactively.'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.coefficients_file):
        print(f"Error: File not found: {args.coefficients_file}")
        sys.exit(1)
    
    # Run test
    try:
        df_results, fig = test_wind_statistics(
            coefficients_file=args.coefficients_file,
            furt_archive_path=args.furt_path,
            sampling_rate=args.sampling_rate,
            n_example_windows=args.n_examples,
            output_plot=args.output_plot
        )
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
