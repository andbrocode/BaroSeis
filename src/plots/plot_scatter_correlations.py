"""
Module for plotting scatter plots of seismic data vs pressure data correlations.
"""

from typing import Dict, Optional, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, UTCDateTime
import pandas as pd
from scipy import stats

def plot_scatter_correlations(bs, time_unit: str = "minutes", 
                            channel_type: str = "A", out: bool = False,
                            figsize: Tuple[int, int] = (16, 12),
                            alpha: float = 0.6,
                            s: float = 1.0) -> Optional[plt.Figure]:
    """
    Plot scatter plots of seismic data (Z, N, E) vs pressure data (p, h, dh, dp).
    
    Creates a 3x4 grid showing:
    - Rows: Z, N, E components
    - Columns: p (pressure), h (hilbert), dh (hilbert derivative), dp (pressure derivative)
    
    Args:
        bs: baroseis instance with loaded and processed data
        time_unit: Time unit for x-axis ('minutes', 'hours', 'days', 'seconds')
        channel_type: Type of channel to plot:
                     'J' for rotation rate
                     'A' for tilt
                     'H' for acceleration
        out: If True, return figure object
        figsize: Figure size (width, height)
        alpha: Transparency for scatter points
        s: Size of scatter points
        
    Returns:
        matplotlib Figure object if out=True, otherwise None
    """
    
    # Set units based on channel type
    if channel_type == 'J':
        ylabel = "Rotation Rate (nrad/s)"
        yscale = 1e9
        coef_unit = "nrad/s/hPa"
    elif channel_type == 'A':
        ylabel = "Tilt (nrad)"
        yscale = 1e9
        coef_unit = "nrad/hPa"
    elif channel_type == 'H':
        ylabel = "Acceleration (nm/s²)"
        yscale = 1e9
        coef_unit = "nm/s²/hPa"
    else:
        ylabel = "Amplitude"
        yscale = 1.0
        coef_unit = "units/hPa"
    
    # Set time scaling
    tscale_dict = {
        "hours": 1/3600,
        "days": 1/86400,
        "minutes": 1/60,
        "seconds": 1
    }
    tscale = tscale_dict.get(time_unit, 1/60)
    
    # Create figure and subplots
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle(f'Seismic vs Pressure Correlations - {channel_type} Components', 
                 fontsize=14, fontweight='bold')
    
    fontsize = 12

    # Define components and pressure types
    components = ['Z', 'N', 'E']
    pressure_types = ['p', 'h', 'dh', 'dp']
    pressure_labels = ['P (Pa)', 'H (Pa)', '$\partial_t$H (Pa/s)', '$\partial_t$P (Pa/s)']
    
    # Get pressure data
    try:
        tr_p = bs.st.select(channel="*D*")[0]  # Pressure
        tr_h = bs.st.select(channel="*DH")[0]  # Hilbert transform
    except IndexError:
        raise ValueError("Could not find pressure or Hilbert transform traces")
    
    # Calculate derivatives if not already present
    if not any('DP' in tr.id for tr in bs.st):
        # Calculate pressure derivative
        tr_dp = tr_p.copy()
        tr_dp.data = np.gradient(tr_p.data, tr_p.stats.delta)
        tr_dp.stats.channel = tr_dp.stats.channel.replace('D', 'DP')
        bs.st += tr_dp
    
    if not any('DH' in tr.id for tr in bs.st):
        # Calculate Hilbert derivative
        tr_dh = tr_h.copy()
        tr_dh.data = np.gradient(tr_h.data, tr_h.stats.delta)
        tr_dh.stats.channel = tr_dh.stats.channel.replace('H', 'DH')
        bs.st += tr_dh
    
    # Get derivative traces
    try:
        tr_dp = bs.st.select(channel="*DP")[0]  # Pressure derivative
        tr_dh = bs.st.select(channel="*DH")[0]  # Hilbert derivative
    except IndexError:
        # Calculate derivatives on the fly
        tr_dp = tr_p.copy()
        tr_dp.data = np.gradient(tr_p.data, tr_p.stats.delta)
        tr_dh = tr_h.copy()
        tr_dh.data = np.gradient(tr_h.data, tr_h.stats.delta)
    
    # Prepare pressure data arrays
    pressure_data = {
        'p': tr_p.data,
        'h': tr_h.data,
        'dh': tr_dh.data,
        'dp': tr_dp.data
    }
    
    # Process each component
    for i, comp in enumerate(components):
        # Get seismic data for this component
        try:
            tr_seis = bs.st.select(channel=f"*{comp}")[0]
            seis_data = tr_seis.data * yscale  # Convert to appropriate units
        except IndexError:
            print(f"Warning: Could not find seismic data for component {comp}")
            continue
        
        # Process each pressure type
        for j, (p_type, p_label) in enumerate(zip(pressure_types, pressure_labels)):
            ax = axes[i, j]
            
            # Get pressure data
            p_data = pressure_data[p_type]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(seis_data) | np.isnan(p_data))
            if not np.any(valid_mask):
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=fontsize)
                ax.set_xlabel(f'{p_label}')
                ax.set_ylabel(f'{comp} {ylabel}')
                continue
            
            x_data = p_data[valid_mask]
            y_data = seis_data[valid_mask]
            
            # Create scatter plot
            ax.scatter(x_data, y_data, alpha=alpha, s=s, c='black', edgecolors='none')
            
            # Calculate correlation coefficient
            if len(x_data) > 1:
                corr_coef, p_value = stats.pearsonr(x_data, y_data)
                
                # Add correlation info to plot
                ax.text(0.05, 0.95, f'CC = {corr_coef:.1f}', 
                       transform=ax.transAxes, fontsize=fontsize-2, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')
                
                # Add trend line
                if not np.isnan(corr_coef):
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Set labels and title
            ax.set_xlabel(f'{p_label}', fontsize=fontsize)
            ax.set_ylabel(f'{comp} {ylabel}', fontsize=fontsize)
            ax.grid(True, alpha=0.3)
            

    plt.tight_layout()
    
    if out:
        return fig
    else:
        plt.show()
        return None
