"""
Module for plotting continuous wavelet transform (CWT) results.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def plot_cwt(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Plot continuous wavelet transform results for barometer and seismometer/rotation data.
    
    This function creates a multi-panel plot showing the CWT scalograms for the
    barometer and each seismometer/rotation component. The scalograms show the
    signal's frequency content over time.
    
    Args:
        results: Dictionary containing CWT results.
            Expected structure:
            {
                'barometer': {
                    'time': array of times,
                    'scales': array of scales,
                    'cwt': 2D array of CWT coefficients,
                    'frequencies': array of frequencies
                },
                'component_id': {
                    'time': array of times,
                    'scales': array of scales,
                    'cwt': 2D array of CWT coefficients,
                    'frequencies': array of frequencies
                }
            }
    
    Example:
        >>> cwt_results = bs.compute_cwt()
        >>> plot_cwt(cwt_results)
    """
    # Determine number of components
    n_components = len(results)
    
    # Create figure
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components),
                            sharex=True)
    if n_components == 1:
        axes = [axes]
    
    # Plot each component
    for i, (comp_id, data) in enumerate(results.items()):
        # Get data
        time = data['time']
        freqs = data['frequencies']
        cwt = np.abs(data['cwt'])
        
        # Create time-frequency mesh for pcolormesh
        time_mesh, freq_mesh = np.meshgrid(time, freqs)
        
        # Plot scalogram
        im = axes[i].pcolormesh(time_mesh, freq_mesh, cwt,
                               cmap='viridis', shading='auto')
        
        # Format subplot
        axes[i].set_ylabel('Frequency (Hz)')
        axes[i].set_yscale('log')
        axes[i].set_title(f'CWT - {comp_id}')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Amplitude')
    
    # Format overall plot
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
