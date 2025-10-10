"""
Plotting module for baroseis package.

This module contains functions for plotting various aspects of barometer and seismometer data:
- Coherence plots
- Cross-correlation plots
- Continuous wavelet transform plots
- Waveform plots
- Residual plots
- Spectral comparison plots
"""

from .plot_coherence import plot_coherence
from .plot_cross_correlations import plot_cross_correlations
from .plot_cwt import plot_cwt
from .plot_waveforms import plot_waveforms
from .plot_residuals import plot_residuals
from .plot_spectra import compare_spectra

__all__ = [
    'plot_coherence',
    'plot_cross_correlations',
    'plot_cwt',
    'plot_waveforms',
    'plot_residuals',
    'compare_spectra',
]