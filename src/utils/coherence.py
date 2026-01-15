#!/usr/bin/python
"""
Coherence computation utility module.

This module provides functions for computing coherence between two data arrays
for multiple time windows and overlaps.

Author: Andreas Brotzer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, savgol_filter
from typing import Union, List, Tuple, Optional, Dict, Any


def compute_coherence_windows(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    window_sec: Union[float, List[float]],
    overlap: Union[float, List[float]] = 0.5,
    plot: bool = False,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    smooth: Optional[Union[int, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Compute coherence between two data arrays for given time windows and overlaps.
    
    This function computes coherence for one or multiple window/overlap combinations
    and optionally plots the results with colored lines using the viridis colormap by default.
    
    Args:
        data1: First data array
        data2: Second data array
        fs: Sampling frequency in Hz
        window_sec: Window length(s) in seconds. Can be a single float or list of floats.
        overlap: Overlap fraction(s) (0-1). Can be a single float or list of floats.
                 Default is 0.5 (50% overlap).
        plot: If True, plot the coherence results. Default is False.
        colors: List of colors for plotting. If None, uses viridis colormap by default.
        labels: List of labels for each window/overlap combination. If None, auto-generates.
        figsize: Figure size tuple (width, height) in inches. Default is (10, 6).
        fmin: Minimum frequency to plot (Hz). If None, uses minimum available frequency.
        fmax: Maximum frequency to plot (Hz). If None, uses maximum available frequency.
        ax: Matplotlib axes object to plot on. If None, creates new figure.
        smooth: Optional smoothing of coherence data. If None, no smoothing is applied.
               If an integer, applies moving average with that window size (must be odd).
               If a dict, can specify:
                   - 'method': 'moving_avg' or 'savgol'
                   - 'window': window size (must be odd for savgol)
                   - 'polyorder': polynomial order for savgol (default: 2)
               Smoothing preserves all data points (no downsampling).
    
    Returns:
        Dictionary containing:
            - 'results': List of dictionaries, each containing:
                - 'window_sec': Window length used
                - 'overlap': Overlap fraction used
                - 'frequencies': Array of frequencies
                - 'coherence': Array of coherence values (smoothed if smooth parameter used)
            - 'figure': Matplotlib figure object (if plot=True), None otherwise

    """
    # Validate inputs
    if len(data1) != len(data2):
        raise ValueError("data1 and data2 must have the same length")
    
    if len(data1) == 0:
        raise ValueError("Data arrays cannot be empty")
    
    # Convert single values to lists
    if isinstance(window_sec, (int, float)):
        window_sec = [window_sec]
    if isinstance(overlap, (int, float)):
        overlap = [overlap]
    
    # Validate window and overlap values
    for w in window_sec:
        if w <= 0:
            raise ValueError("window_sec must be positive")
        npts = int(w * fs)
        if npts > len(data1):
            raise ValueError(f"Window length ({w}s) longer than data duration ({len(data1)/fs:.1f}s)")
    
    for o in overlap:
        if not 0 <= o < 1:
            raise ValueError("overlap must be between 0 and 1")
    
    # Prepare colors and labels
    n_combinations = len(window_sec) * len(overlap)
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_combinations))
    elif len(colors) < n_combinations:
        # Repeat colors if not enough provided
        colors = (colors * ((n_combinations // len(colors)) + 1))[:n_combinations]
    
    if labels is None:
        labels = []
        for w in window_sec:
            for o in overlap:
                labels.append(f"Window={w:.0f}s, Overlap={o:.1%}")
    elif len(labels) < n_combinations:
        # Extend labels if not enough provided
        labels = labels + [f"Config {i+1}" for i in range(len(labels), n_combinations)]
    
    # Compute coherence for each combination
    results = []
    color_idx = 0
    
    for w in window_sec:
        for o in overlap:
            npts = int(w * fs)
            noverlap = int(npts * o)
            
            try:
                # Compute coherence
                freq, coh = coherence(
                    data1, data2,
                    fs=fs,
                    nperseg=npts,
                    noverlap=noverlap,
                    window='hann'
                )
                
                # Apply smoothing if requested (preserves all data points)
                if smooth is not None:
                    if isinstance(smooth, int):
                        # Simple moving average
                        window_size = smooth
                        if window_size % 2 == 0:
                            window_size += 1  # Make odd for symmetric smoothing
                        if window_size > len(coh):
                            window_size = len(coh) if len(coh) % 2 == 1 else len(coh) - 1
                        if window_size >= 3:
                            # Use convolution for moving average
                            kernel = np.ones(window_size) / window_size
                            coh = np.convolve(coh, kernel, mode='same')
                    elif isinstance(smooth, dict):
                        method = smooth.get('method', 'savgol')
                        window = smooth.get('window', 31)
                        
                        if method == 'savgol':
                            polyorder = smooth.get('polyorder', 2)
                            if window % 2 == 0:
                                window += 1  # Savitzky-Golay requires odd window
                            if window > len(coh):
                                window = len(coh) if len(coh) % 2 == 1 else len(coh) - 1
                            if window >= 3 and window > polyorder:
                                coh = savgol_filter(coh, window, polyorder)
                        elif method == 'moving_avg':
                            if window % 2 == 0:
                                window += 1
                            if window > len(coh):
                                window = len(coh) if len(coh) % 2 == 1 else len(coh) - 1
                            if window >= 3:
                                kernel = np.ones(window) / window
                                coh = np.convolve(coh, kernel, mode='same')
                
                results.append({
                    'window_sec': w,
                    'overlap': o,
                    'frequencies': freq,
                    'coherence': coh,
                    'color': colors[color_idx],
                    'label': labels[color_idx]
                })
                
                color_idx += 1
                
            except Exception as e:
                print(f"Warning: Error computing coherence for window={w:.0f}s, overlap={o:.1%}: {str(e)}")
                continue
    
    if not results:
        raise ValueError("No valid coherence results computed")
    
    # Plot if requested
    fig = None
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        for result in results:
            freq = result['frequencies']
            coh = result['coherence']
            
            # Apply frequency limits
            if fmin is not None or fmax is not None:
                mask = np.ones_like(freq, dtype=bool)
                if fmin is not None:
                    mask &= freq >= fmin
                if fmax is not None:
                    mask &= freq <= fmax
                freq = freq[mask]
                coh = coh[mask]
            
            # Plot coherence
            ax.semilogx(freq, coh, 
                       color=result['color'],
                       label=result['label'],
                       alpha=0.7,
                       linewidth=1.5)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title('Coherence between Two Signals')
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim(0, 1.01)
        
        if fmin is not None:
            ax.set_xlim(left=fmin)
        if fmax is not None:
            ax.set_xlim(right=fmax)
        
        plt.tight_layout()
        
        if ax is None:  # Only show if we created the figure
            plt.show()
    
    return {
        'results': results,
        'figure': fig
    }
