"""
Module for plotting coherence between barometer and seismometer/rotation components.
"""

from typing import Dict, Optional, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

def plot_coherence(coherence_dict: Dict[str, Dict[str, Any]], fmin: Optional[float] = None,
                  fmax: Optional[float] = None, figsize: Tuple[int, int] = (10,6), 
                  out: bool = False) -> Optional[plt.Figure]:
    """
    Plot coherence results between barometer and seismometer/rotation components.
    
    Args:
        coherence_dict: Dictionary containing coherence results from compute_coherence().
            Expected structure:
            {
                'trace_id': {
                    'frequencies': array of frequencies,
                    'coherence': array of coherence values,
                    'window_sec': window length used,
                    'overlap': overlap fraction used,
                    'baro_id': ID of barometer channel used
                }
            }
        fmin: Minimum frequency to plot (Hz). If None, uses minimum available frequency.
        fmax: Maximum frequency to plot (Hz). If None, uses maximum available frequency.
        figsize: Figure size tuple (width, height) in inches.
        out: If True, return the figure object instead of displaying it.
    
    Returns:
        matplotlib.figure.Figure if out=True, None otherwise.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get barometer ID for title
    baro_id = list(coherence_dict.values())[0]['baro_id']
    
    for component_id, data in coherence_dict.items():
        freq = data['frequencies']
        coh = data['coherence']
        
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
        ax.semilogx(freq, coh, label=component_id, alpha=0.7)
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    ax.set_title(f'Coherence: {baro_id} vs Seismic Components')
    ax.legend(loc='best')
    ax.set_ylim(0, 1.01)
    
    if fmin is not None:
        ax.set_xlim(left=fmin)
    if fmax is not None:
        ax.set_xlim(right=fmax)
    
    plt.tight_layout()
    plt.show()

    if out:
        return fig