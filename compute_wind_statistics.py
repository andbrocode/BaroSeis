#!/usr/bin/python3
"""
Script to compute wind statistics for time windows from coefficients CSV file.

This script:
1. Loads a coefficients CSV file with time windows
2. Loads FURT weather station data for the time period
3. Computes wind speed and direction statistics for each window
4. Saves results to a CSV file

Author: Andreas Brotzer
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from obspy import UTCDateTime, Stream, Trace
from numpy import arange, ones, nan, array
from pandas import concat, to_datetime, read_csv, DataFrame
from tqdm import tqdm


def load_furt_stream(starttime, endtime, show_raw=False, sampling_rate=1.0, 
                       path_to_archive='/bay200/gif_online/FURT/WETTER/'):
    """
    Load a selection of data of FURT weather station for certain times and return an obspy stream.
    
    Parameters:
        starttime: Start time (UTCDateTime or string)
        endtime: End time (UTCDateTime or string)
        show_raw: bool (True/False) -> shows raw data FURT head
        sampling_rate: Sampling rate in Hz (default: 1.0)
        path_to_archive: Path to FURT weather station archive
    
    Returns:
        obspy.Stream: Stream containing FURT weather data
    """
    
    def _add_trace(cha, tbeg, dat, dt=1):
        """Helper function to create an ObsPy Trace."""
        tr = Trace()
        tr.stats.station = 'FURT'
        tr.stats.network = 'BW'
        tr.stats.channel = str(cha)
        tr.stats.sampling_rate = 1/dt
        tr.stats.starttime = UTCDateTime(tbeg)
        tr.data = array(dat)
        return tr

    def _resample(df, freq='1s'):
        """Helper function to resample dataframe."""
        # Check for NaN in dates
        if df.date.isna().any():
            print(" -> NaN values found and removed from column date")
            df = df.dropna(axis=0, subset=["date"])
            try:
                df["date"] = df["date"].astype(int)
            except:
                df["date"] = df["Date"].astype(int)
        
        # Make column with datetime
        df['datetime'] = df['date'].astype(str).str.rjust(6,"0")+" "+df['time'].astype(str).str.rjust(6,"0")

        # Drop datetime duplicates
        df = df[df.duplicated("datetime", keep="first") != True]

        # Convert to pandas datetime object
        df['datetime'] = to_datetime(df['datetime'], format="%d%m%y %H%M%S", errors="ignore")

        # Set datetime column as index
        df.set_index('datetime', inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated()]

        # Resample
        df = df.asfreq(freq=freq)

        return df

    starttime = UTCDateTime(starttime)
    endtime = UTCDateTime(endtime)

    output_text = []
    new_delta = 1/sampling_rate

    if not Path(path_to_archive).exists():
        output_text.append(f"  -> Path: {path_to_archive}, does not exist!")
        print(f"  -> Path: {path_to_archive}, does not exists!")

    # Declare empty dataframe
    df = DataFrame()

    for i, date in enumerate(arange(starttime.date, (endtime+86400+10).date)):
        date = UTCDateTime(str(date)).date
        filename = f'FURT.WSX.D.{str(date.day).rjust(2,"0")}{str(date.month).rjust(2,"0")}{str(date.year).rjust(2,"0")[-2:]}.0000'

        if show_raw:
            df0 = read_csv(path_to_archive+filename)
            print(df0.columns.tolist())

        try:
            try:
                try:
                    df0 = read_csv(path_to_archive+filename, header=0, usecols=[0,1,5,8,10,12,13,14], 
                                   names=['date', 'time', 'Dm', 'Sm', 'T', 'H', 'P','Rc'])
                except:
                    df0 = read_csv(path_to_archive+filename, usecols=[0,1,5,8,10,12,13,14], 
                                   names=['date', 'time', 'Dm', 'Sm', 'T', 'H', 'P','Rc'])
            except:
                print(f" -> loading of {filename} failed!")
                continue

            # Substitute strings with floats
            # Air temperature Ta in degree C
            TT = ones(len(df0['T']))*nan
            for _n, t in enumerate(df0['T']):
                try:
                    TT[_n] = float(str(str(t).split("=")[1]).split("C")[0])
                except:
                    continue
            df0['T'] = TT

            # Air pressure Pa in hPa
            PP = ones(len(df0['P']))*nan
            for _n, p in enumerate(df0['P']):
                try:
                    PP[_n] = float(str(str(p).split("=")[1]).split("H")[0])
                except:
                    continue
            df0['P'] = PP

            # Relative humidity Ua in %RH
            HH = ones(len(df0['H']))*nan
            for _n, h in enumerate(df0['H']):
                try:
                    HH[_n] = float(str(str(h).split("=")[1]).split("P")[0])
                except:
                    continue
            df0['H'] = HH

            # Rain accumulation in mm
            Rc = ones(len(df0['Rc']))*nan
            for _n, rc in enumerate(df0['Rc']):
                try:
                    Rc[_n] = float(str(str(rc).split("=")[1]).split("M")[0])
                except:
                    continue
            df0['Rc'] = Rc

            # Wind speed average in m/s
            Sm = ones(len(df0['Sm']))*nan
            for _n, sm in enumerate(df0['Sm']):
                try:
                    Sm[_n] = float(str(str(sm).split("=")[1]).split("M")[0])
                except:
                    continue
            df0['Sm'] = Sm

            # Wind direction average in degrees
            Dm = ones(len(df0['Dm']))*nan
            for _n, dm in enumerate(df0['Dm']):
                try:
                    Dm[_n] = float(str(str(dm).split("=")[1]).split("D")[0])
                except:
                    continue
            df0['Dm'] = Dm

            if df.empty:
                df = df0
            else:
                try:
                    df = concat([df, df0])
                except:
                    print(f"  -> failed to concat for {filename}")
        except Exception as e:
            print(e)
            output_text.append(f"  -> {filename}, failed!")

    # Reset the index for the joined frame
    df.reset_index(inplace=True, drop=True)

    # Resample dataframe and avoid data gaps
    try:
        df = _resample(df, freq=f'{new_delta}S')
    except Exception as e:
        print(e)

    for text in output_text:
        print(text)

    df_starttime = UTCDateTime(df.index[0])

    # Create stream and attach traces
    st0 = Stream()
    st0 += _add_trace("LAT", df_starttime, df['T'], dt=new_delta)
    st0 += _add_trace("LAP", df_starttime, df['P'], dt=new_delta)
    st0 += _add_trace("LAH", df_starttime, df['H'], dt=new_delta)
    st0 += _add_trace("LAR", df_starttime, df['Rc'], dt=new_delta)
    st0 += _add_trace("LAW", df_starttime, df['Sm'], dt=new_delta)
    st0 += _add_trace("LAD", df_starttime, df['Dm'], dt=new_delta)

    # Trim to specified time period
    st0.trim(starttime, endtime-new_delta/2)

    t1, t2 = endtime-new_delta, st0.select(channel='*T')[0].stats.endtime
    if t1 != t2:
        print(f"Specified end: {t1} \nTrace end:     {t2}")

    return st0


def circular_mean(angles, degrees=True):
    """
    Compute circular mean of angles.
    
    Args:
        angles: Array of angles
        degrees: If True, angles are in degrees; if False, in radians
    
    Returns:
        Circular mean in same units as input
    """
    if degrees:
        angles = np.deg2rad(angles)
    
    # Remove NaN values
    angles = angles[~np.isnan(angles)]
    
    if len(angles) == 0:
        return np.nan
    
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    
    mean_angle = np.arctan2(mean_sin, mean_cos)
    
    if degrees:
        mean_angle = np.rad2deg(mean_angle)
        # Convert to 0-360 range
        if mean_angle < 0:
            mean_angle += 360
    
    return mean_angle


def circular_median(angles, degrees=True):
    """
    Compute circular median of angles.
    
    Args:
        angles: Array of angles
        degrees: If True, angles are in degrees; if False, in radians
    
    Returns:
        Circular median in same units as input
    """
    if degrees:
        angles = np.deg2rad(angles)
    
    # Remove NaN values
    angles = angles[~np.isnan(angles)]
    
    if len(angles) == 0:
        return np.nan
    
    # For circular median, we can use the circular mean as an approximation
    # or compute the angle that minimizes the sum of circular distances
    # Simple approach: use circular mean
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    
    median_angle = np.arctan2(mean_sin, mean_cos)
    
    if degrees:
        median_angle = np.rad2deg(median_angle)
        # Convert to 0-360 range
        if median_angle < 0:
            median_angle += 360
    
    return median_angle


def circular_std(angles, degrees=True):
    """
    Compute circular standard deviation of angles.
    
    Args:
        angles: Array of angles
        degrees: If True, angles are in degrees; if False, in radians
    
    Returns:
        Circular standard deviation in same units as input
    """
    if degrees:
        angles = np.deg2rad(angles)
    
    # Remove NaN values
    angles = angles[~np.isnan(angles)]
    
    if len(angles) == 0:
        return np.nan
    
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    
    # Circular standard deviation
    std = np.sqrt(-2 * np.log(R))
    
    if degrees:
        std = np.rad2deg(std)
    
    return std


def compute_wind_statistics(
    coefficients_file: str,
    output_file: str = None,
    furt_archive_path: str = '/import/bay200/gif_online/FURT/WETTER/',
    sampling_rate: float = 1.0
):
    """
    Compute wind statistics for time windows in coefficients CSV file.
    
    Args:
        coefficients_file: Path to coefficients CSV file
        output_file: Path to output CSV file. If None, uses same directory as input.
        furt_archive_path: Path to FURT weather station archive
        sampling_rate: Sampling rate for FURT data (Hz)
    
    Returns:
        DataFrame with wind statistics
    """
    # Load coefficients CSV
    print(f"Loading coefficients file: {coefficients_file}")
    df_coeff = pd.read_csv(coefficients_file)
    
    # Check required columns
    if 'window_start' not in df_coeff.columns or 'window_end' not in df_coeff.columns:
        raise ValueError("CSV file must contain 'window_start' and 'window_end' columns")
    
    # Convert to datetime
    df_coeff['window_start'] = pd.to_datetime(df_coeff['window_start'])
    df_coeff['window_end'] = pd.to_datetime(df_coeff['window_end'])
    
    # Get overall time range
    overall_start = df_coeff['window_start'].min()
    overall_end = df_coeff['window_end'].max()
    
    print(f"Time range: {overall_start} to {overall_end}")
    print(f"Number of windows: {len(df_coeff)}")
    print(f"\nLoading FURT data per window (with day-level caching)...")
    
    # Initialize results list and cache for daily data
    results = []
    daily_cache = {}  # Cache key: date string (YYYY-MM-DD), value: Stream
    
    # Process each window with progress bar
    print(f"\nComputing statistics for {len(df_coeff)} windows...")
    for idx, row in tqdm(df_coeff.iterrows(), total=len(df_coeff), desc="Processing windows", unit="window"):
        window_start = UTCDateTime(row['window_start'])
        window_end = UTCDateTime(row['window_end'])
        
        # Initialize result row with NaN values
        result = {
            'window_start': row['window_start'],
            'window_end': row['window_end'],
            'wind_speed_mean': np.nan,
            'wind_speed_median': np.nan,
            'wind_speed_max': np.nan,
            'wind_dir_mean': np.nan,
            'wind_dir_median': np.nan,
            'wind_dir_std': np.nan,
        }
        
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
                # Load data for this day (with small buffer to handle edge cases)
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
                    tqdm.write(f"Warning: Error loading FURT data for {date_key}: {e}")
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
                tqdm.write(f"Warning: No wind data available for window {idx+1} ({window_start} to {window_end})")
                results.append(result)
                continue
            
            try:
                tr_wind_speed = st_window.select(channel='LAW').merge(method=1)[0]
                tr_wind_dir = st_window.select(channel='LAD').merge(method=1)[0]
            except (IndexError, Exception) as e:
                tqdm.write(f"Warning: Could not extract wind traces for window {idx+1}: {e}")
                results.append(result)
                continue
            
            # Trim to exact window
            tr_ws_win = tr_wind_speed.copy()
            tr_wd_win = tr_wind_dir.copy()
            tr_ws_win.trim(window_start, window_end)
            tr_wd_win.trim(window_start, window_end)
            
            # Get data arrays
            wind_speed = tr_ws_win.data
            wind_dir = tr_wd_win.data
            
            # Remove NaN values for statistics
            wind_speed_clean = wind_speed[~np.isnan(wind_speed)]
            wind_dir_clean = wind_dir[~np.isnan(wind_dir)]
            
            # Compute wind speed statistics
            if len(wind_speed_clean) > 0:
                result['wind_speed_mean'] = np.round(np.mean(wind_speed_clean), 1)
                result['wind_speed_median'] = np.round(np.median(wind_speed_clean), 1)
                result['wind_speed_max'] = np.round(np.max(wind_speed_clean), 1)
            
            # Compute wind direction statistics (circular)
            if len(wind_dir_clean) > 0:
                # Ensure angles are in 0-360 range
                wind_dir_clean = wind_dir_clean % 360
                
                result['wind_dir_mean'] = np.round(circular_mean(wind_dir_clean, degrees=True), 0)
                result['wind_dir_median'] = np.round(circular_median(wind_dir_clean, degrees=True), 0)
                result['wind_dir_std'] = np.round(circular_std(wind_dir_clean, degrees=True), 0)
            
        except Exception as e:
            tqdm.write(f"Warning: Error processing window {idx+1} ({window_start} to {window_end}): {e}")
        
        results.append(result)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Set output file path
    if output_file is None:
        coeff_path = Path(coefficients_file)
        output_file = coeff_path.parent / f"{coeff_path.stem}_wind_stats.csv"
    
    # Save to CSV
    print(f"\nSaving results to: {output_file}")
    df_results.to_csv(output_file, index=False)
    
    print(f"Done! Processed {len(df_results)} windows.")
    print(f"\nStatistics summary:")
    print(f"  Wind speed mean: {df_results['wind_speed_mean'].mean():.2f} m/s")
    print(f"  Wind speed max: {df_results['wind_speed_max'].max():.2f} m/s")
    print(f"  Wind direction mean: {df_results['wind_dir_mean'].mean():.1f}°")
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute wind statistics for time windows in coefficients CSV file"
    )
    parser.add_argument(
        'coefficients_file',
        type=str,
        help='Path to coefficients CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: same directory as input with _wind_stats suffix)'
    )
    parser.add_argument(
        '--furt-path',
        type=str,
        default='/bay200/gif_online/FURT/WETTER/',
        help='Path to FURT weather station archive'
    )
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=1.0,
        help='Sampling rate for FURT data (Hz, default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.coefficients_file):
        print(f"Error: File not found: {args.coefficients_file}")
        sys.exit(1)
    
    # Run computation
    try:
        df_results = compute_wind_statistics(
            coefficients_file=args.coefficients_file,
            output_file=args.output,
            furt_archive_path=args.furt_path,
            sampling_rate=args.sampling_rate
            
        )
        print("\nSuccess!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
