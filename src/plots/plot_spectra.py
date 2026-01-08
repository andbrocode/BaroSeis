"""
Module for comparing spectra between observed and predicted/residual data.
"""

from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream
from scipy.signal import welch
from scipy.fft import fft, fftfreq, fftshift

def compare_spectra(st: Stream, config: Dict[str, Any], method: str = 'fft',
                   channel_type: str = 'J', fmin: float = 0.0005, fmax: float = 0.1,
                   window: str = 'hann', log_scale: bool = True, db_scale: bool = False,
                   figsize: Tuple[int, int] = (12, 10), compare_residual: bool = True) -> plt.Figure:
    """
    Compare spectra between observed data and either predicted data or residuals.
    
    This function creates a multi-panel plot showing the spectral comparison
    between original data and either predicted data or residuals for each
    component.
    
    Args:
        st: ObsPy Stream containing the data
        config: Configuration dictionary
        method: Method to compute spectra ('fft' or 'welch')
        channel_type: Channel type to use:
            'J' for rotation rate
            'A' for tilt
            'H' for acceleration
        fmin: Minimum frequency to display (Hz)
        fmax: Maximum frequency to display (Hz)
        window: Window function for FFT ('hann', 'hamming', 'blackman', etc.)
        log_scale: Whether to use logarithmic scale for frequency axis
        db_scale: Whether to show amplitudes in decibels (20*log10(amplitude))
        figsize: Figure size tuple (width, height) in inches
        compare_residual: If True, compare observed vs residual
                        If False, compare observed vs predicted
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    
    Example:
        >>> fig = compare_spectra(st, config, method='welch', channel_type='J')
        >>> plt.show()
    """
    # Set scaling factors based on channel type
    if channel_type == "J":
        yscale = 1e9
        ylabel = "Rotation Rate (nrad/s)"
    elif channel_type == "A":
        yscale = 1e9
        ylabel = "Tilt (nrad)"
    else:  # channel_type == "H"
        yscale = 1e9
        ylabel = "Acceleration (nm/s²)"
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot each component
    for i, comp in enumerate(['N', 'E', 'Z']):
        try:
            # Get original and predicted data
            tr_orig = st.select(channel=f"*{channel_type}{comp}", location="").copy()[0]
            tr_pred = st.select(channel=f"*{channel_type}{comp}", location="PP").copy()[0]
            
            # Calculate residual if needed
            if compare_residual:
                tr_comp = tr_orig.copy()
                tr_comp.data = tr_orig.data - tr_pred.data
                comp_label = 'Residual'
            else:
                tr_comp = tr_pred
                comp_label = 'Predicted'
            
            # Compute spectra based on method
            if method.lower() == 'fft':
                # FFT method
                from scipy.fft import fft, fftfreq
                
                # Original data
                n = len(tr_orig.data)
                win = eval(f"np.{window}(n)")
                spec_orig = fft(tr_orig.data * win)
                freq = fftfreq(n, d=tr_orig.stats.delta)
                
                # Comparison data
                spec_comp = fft(tr_comp.data * win)
                
                # Get positive frequencies
                pos_freq = fftshift(freq)
                mag_orig = fftshift(np.abs(spec_orig) * tr_orig.delta / (2*np.pi)) * yscale
                mag_comp = fftshift(np.abs(spec_comp) * tr_comp.delta / (2*np.pi)) * yscale
                # pos_freq = freq[0:n//2]
                # mag_orig = np.abs(spec_orig[0:n//2]) * 2.0/n * yscale
                # mag_comp = np.abs(spec_comp[0:n//2]) * 2.0/n * yscale
                
            else:  # method == 'welch'
                # Welch's method
                nperseg = int(tr_orig.stats.sampling_rate * 3600)  # 1-hour segments
                noverlap = nperseg // 2
                
                # Original data
                freq, psd_orig = welch(tr_orig.data, fs=tr_orig.stats.sampling_rate,
                                     window=window, nperseg=nperseg, noverlap=noverlap)
                mag_orig = np.sqrt(psd_orig) * yscale
                
                # Comparison data
                _, psd_comp = welch(tr_comp.data, fs=tr_comp.stats.sampling_rate,
                                  window=window, nperseg=nperseg, noverlap=noverlap)
                mag_comp = np.sqrt(psd_comp) * yscale
                pos_freq = freq
            
            # Convert to dB if requested
            if db_scale:
                mag_orig = 20 * np.log10(mag_orig)
                mag_comp = 20 * np.log10(mag_comp)
                ylabel = ylabel + " (dB)"
            
            # Apply frequency limits
            mask = (pos_freq >= fmin) & (pos_freq <= fmax)
            pos_freq = pos_freq[mask]
            mag_orig = mag_orig[mask]
            mag_comp = mag_comp[mask]
            
            # Plot spectra
            axes[i].plot(pos_freq, mag_orig, 'k-', label='Original', alpha=0.7)
            axes[i].plot(pos_freq, mag_comp, 'r-', label=comp_label, alpha=0.7)
            
            # Format subplot
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f'{comp}-component')
            if log_scale:
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
        except Exception as e:
            print(f"Could not plot component {comp}: {str(e)}")
    
    # Set title
    title = (f"{config['tbeg'].date} {str(config['tbeg'].time).split('.')[0]} - "
             f"{str(config['tend'].time).split('.')[0]} UTC")
    if 'fmin' in config and 'fmax' in config:
        title += f"\nf = {config['fmin']*1e3:.1f} - {config['fmax']*1e3:.1f} mHz"
    fig.suptitle(title)
    
    # Set x-axis label
    axes[-1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    return fig
