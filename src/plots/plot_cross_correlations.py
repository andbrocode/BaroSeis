"""
Module for plotting cross-correlations between barometer and seismometer/rotation components.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def plot_cross_correlations(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Plot cross-correlation results between barometer and seismometer/rotation components.
    
    This function creates a multi-panel plot showing:
    1. Cross-correlation functions over time
    2. Time shifts over time
    3. Maximum correlation coefficients over time
    
    Args:
        results: Dictionary containing cross-correlation results.
            Expected structure:
            {
                'channel_id': {
                    'times': array of times,
                    'ccf': array of cross-correlation functions,
                    'lags': array of time lags,
                    'shifts': array of time shifts,
                    'maxima': array of maximum correlation coefficients
                }
            }
    
    Example:
        >>> cc_results = bs.compute_cross_correlation()
        >>> plot_cross_correlations(cc_results)
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 2)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, :])  # Cross-correlation functions
    ax2 = fig.add_subplot(gs[1, 0])  # Time shifts
    ax3 = fig.add_subplot(gs[1, 1])  # Maximum coefficients
    
    # Plot for each component
    colors = {'Z': 'blue', 'N': 'red', 'E': 'green'}
    
    for channel, data in results.items():
        # Get component from channel name
        comp = channel[-1]
        color = colors.get(comp, 'gray')
        
        # Plot cross-correlation functions
        times = data['times']
        lags = data['lags']
        ccf = data['ccf']
        
        # Create time-lag mesh for pcolormesh
        time_mesh, lag_mesh = np.meshgrid(times, lags)
        im = ax1.pcolormesh(time_mesh, lag_mesh, ccf.T, 
                           cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Plot time shifts
        ax2.plot(times, data['shifts'], color=color, 
                label=f'{comp}-component', alpha=0.7)
        
        # Plot maximum coefficients
        ax3.plot(times, data['maxima'], color=color, 
                label=f'{comp}-component', alpha=0.7)
    
    # Format subplots
    # Cross-correlation functions
    ax1.set_ylabel('Time lag (s)')
    ax1.set_title('Cross-correlation Functions')
    plt.colorbar(im, ax=ax1, label='Correlation coefficient')
    
    # Time shifts
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Time shift (s)')
    ax2.set_title('Time Shifts')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Maximum coefficients
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Maximum correlation')
    ax3.set_title('Maximum Correlation Coefficients')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1, 1)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
