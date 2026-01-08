"""
Standalone plot function for comparing spectra between observed and predicted/residual data.
This function can be copy-pasted directly into a Jupyter notebook.

Features:
- Works with ObsPy Stream objects
- Supports FFT and Welch methods
- Optional fractional octave band smoothing
- Configurable frequency range and scaling
"""

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream
from scipy.signal import welch
from scipy.fft import fft, fftfreq


def plot_spectra_standalone(st, config=None, method='fft', channel_type='J', 
                          fmin=0.0005, fmax=0.1, window='hann', log_scale=True, 
                          db_scale=False, figsize=(12, 10), compare_residual=True,
                          smooth_octave=False, octave_fraction=1/3, smooth_method='median'):
    """
    Compare spectra between observed data and either predicted data or residuals.
    
    This function creates a multi-panel plot showing the spectral comparison
    between original data and either predicted data or residuals for each
    component. Can be copy-pasted directly into a Jupyter notebook.
    
    Parameters:
    -----------
    st : obspy.Stream
        ObsPy Stream containing the data
    config : dict, optional
        Configuration dictionary (for title generation)
    method : str, default 'fft'
        Method to compute spectra ('fft' or 'welch')
    channel_type : str, default 'J'
        Channel type to use:
        'J' for rotation rate
        'A' for tilt  
        'H' for acceleration
    fmin : float, default 0.0005
        Minimum frequency to display (Hz)
    fmax : float, default 0.1
        Maximum frequency to display (Hz)
    window : str, default 'hann'
        Window function for FFT ('hann', 'hamming', 'blackman', etc.)
    log_scale : bool, default True
        Whether to use logarithmic scale for frequency axis
    db_scale : bool, default False
        Whether to show amplitudes in decibels (20*log10(amplitude))
    figsize : tuple, default (12, 10)
        Figure size tuple (width, height) in inches
    compare_residual : bool, default True
        If True, compare observed vs residual (observed-predicted)
        If False, compare observed vs predicted
    smooth_octave : bool, default False
        Whether to smooth spectra in fractional octave bands
    octave_fraction : float, default 1/3
        Fraction of octave for smoothing (1/3 = 1/3 octave bands)
    smooth_method : str, default 'median'
        Smoothing method: 'median' or 'mean'
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    
    Example:
    --------
    >>> fig = plot_spectra_standalone(st, method='welch', channel_type='J')
    >>> plt.show()
    """
    
    def smooth_octave_bands(freq, spectrum, octave_fraction=1/3, method='median'):
        """
        Smooth spectrum in fractional octave bands.
        
        Parameters:
        -----------
        freq : array
            Frequency array
        spectrum : array
            Spectrum array
        octave_fraction : float
            Fraction of octave for smoothing
        method : str
            Smoothing method: 'median' or 'mean'
            
        Returns:
        --------
        freq_smooth : array
            Smoothed frequency array
        spectrum_smooth : array
            Smoothed spectrum array
        """
        # Create octave band centers
        f_min = freq[freq > 0].min()
        f_max = freq.max()
        
        # Calculate number of octave bands
        n_octaves = int(np.log2(f_max / f_min) / octave_fraction) + 1
        
        # Create octave band centers
        f_centers = f_min * (2 ** (octave_fraction * np.arange(n_octaves)))
        
        # Calculate band edges
        f_lower = f_centers / (2 ** (octave_fraction / 2))
        f_upper = f_centers * (2 ** (octave_fraction / 2))
        
        # Initialize smoothed arrays
        freq_smooth = []
        spectrum_smooth = []
        
        for i in range(len(f_centers)):
            # Find frequencies within this octave band
            mask = (freq >= f_lower[i]) & (freq <= f_upper[i])
            
            if np.any(mask):
                if method == 'median':
                    smooth_val = np.median(spectrum[mask])
                else:  # mean
                    smooth_val = np.mean(spectrum[mask])
                
                freq_smooth.append(f_centers[i])
                spectrum_smooth.append(smooth_val)
        
        return np.array(freq_smooth), np.array(spectrum_smooth)
    
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
                n = len(tr_orig.data)
                win = eval(f"np.{window}(n)")
                spec_orig = fft(tr_orig.data * win)
                freq = fftfreq(n, d=tr_orig.stats.delta)
                
                # Comparison data
                spec_comp = fft(tr_comp.data * win)
                
                # Get positive frequencies
                pos_freq = freq[0:n//2]
                mag_orig = np.abs(spec_orig[0:n//2]) * 2.0/n * yscale
                mag_comp = np.abs(spec_comp[0:n//2]) * 2.0/n * yscale
                
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
            
            # Apply fractional octave band smoothing if requested
            if smooth_octave:
                pos_freq, mag_orig = smooth_octave_bands(pos_freq, mag_orig, 
                                                        octave_fraction, smooth_method)
                pos_freq, mag_comp = smooth_octave_bands(pos_freq, mag_comp, 
                                                        octave_fraction, smooth_method)
            
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
            axes[i].plot(pos_freq, mag_orig, 'k-', label='Original', alpha=0.7, linewidth=1)
            axes[i].plot(pos_freq, mag_comp, 'r-', label=comp_label, alpha=0.7, linewidth=1)
            
            # Format subplot
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f'{comp}-component')
            if log_scale:
                axes[i].set_xscale('log')
                if not db_scale:  # Only use log scale for y-axis if not in dB
                    axes[i].set_yscale('log')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
        except Exception as e:
            print(f"Could not plot component {comp}: {str(e)}")
    
    # Set title
    if config is not None:
        title = (f"{config['tbeg'].date} {str(config['tbeg'].time).split('.')[0]} - "
                 f"{str(config['tend'].time).split('.')[0]} UTC")
        if 'fmin' in config and 'fmax' in config:
            title += f"\nf = {config['fmin']*1e3:.1f} - {config['fmax']*1e3:.1f} mHz"
        fig.suptitle(title)
    
    # Set x-axis label
    axes[-1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    return fig


# Example usage for copy-pasting into notebook:
"""
# Copy this function into your notebook cell:

def plot_spectra_standalone(st, config=None, method='fft', channel_type='J', 
                          fmin=0.0005, fmax=0.1, window='hann', log_scale=True, 
                          db_scale=False, figsize=(12, 10), compare_residual=True,
                          smooth_octave=False, octave_fraction=1/3, smooth_method='median'):
    # [Function code here - copy the entire function from above]
    pass

# Then use it like this:
fig = plot_spectra_standalone(st, config, method='welch', channel_type='J', 
                            smooth_octave=True, octave_fraction=1/3, smooth_method='median')
plt.show()
"""
