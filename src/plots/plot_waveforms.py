"""
Module for plotting waveforms of rotation and pressure data.
"""

from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream

def plot_waveforms(st: Stream, config: Dict[str, Any], out: bool = False, 
                  time_unit: str = "hours", channel_type: str = "J") -> Optional[plt.Figure]:
    """
    Plot waveforms of rotation and pressure data.
    
    Args:
        st: Stream containing the data
        config: Configuration dictionary
        out: Return figure handle if True
        time_unit: Time unit for x-axis ('hours', 'days', 'minutes', 'seconds')
        channel_type: Channel type to plot ('J' for rotation rate, 'A' for tilt, 'H' for acceleration)
    
    Returns:
        matplotlib.figure.Figure if out=True
    """
    Nrow, Ncol = 5, 1
    yscale = 1e9
    font = 12

    # Set time scaling
    tscale_dict = {
        "hours": 1/3600,
        "days": 1/86400,
        "minutes": 1/60,
        "seconds": 1
    }

    tscale = tscale_dict.get(time_unit)

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 12))

    # Plot rotation components
    for comp, color, idx in zip(['Z', 'N', 'E'], ['tab:blue', 'tab:orange', 'tab:red'], range(3)):
        try:
            tr = st.select(channel=f"*{channel_type}{comp}").copy()[0]  # Select rotation channels
            times = tr.times(reftime=config['tbeg'])*tscale
            data = tr.data*yscale
            
            ax[idx].plot(times, data, label=f"ROMY.{comp}", color=color)
            ax[idx].fill_between(times, 0, data,
                               where=data>0, interpolate=True,
                               color=color, alpha=0.5)
        except Exception as e:
            print(f"Could not plot rotation component {comp}: {str(e)}")

    # Plot pressure and Hilbert transform
    try:
        # Plot original pressure data
        tr_p = st.select(channel="*DO").copy()[0]  # Pressure channel
        times = tr_p.times(reftime=config['tbeg'])*tscale
        data = tr_p.data
        
        ax[3].plot(times, data, label="FFBI.O", color='k')
        ax[3].fill_between(times, 0, data,
                          where=data>0, interpolate=True,
                          color='k', alpha=0.5)
    except Exception as e:
        print(f"Could not plot pressure data: {str(e)}")

    try:
        # Plot Hilbert transform
        tr_h = st.select(channel="*DH").copy()[0]  # Hilbert transform channel
        times = tr_h.times(reftime=config['tbeg'])*tscale
        data = tr_h.data
        
        ax[4].plot(times, data, label="hilbert(FFBI.O)", color='darkgrey')
        ax[4].fill_between(times, 0, data,
                          where=data>0, interpolate=True,
                          color='darkgrey', alpha=0.5)
    except Exception as e:
        print(f"Could not plot Hilbert transform: {str(e)}")

    # Format axes
    for _n in range(Nrow):
        ax[_n].legend(loc=1)
        ax[_n].spines[['right', 'top']].set_visible(False)
        if _n < Nrow-1:
            ax[_n].spines[['bottom']].set_visible(False)

    # Set labels
    for idx in range(3):
        if channel_type == "J":
            ax[idx].set_ylabel("Rotation Rate\n(nrad/s)", fontsize=font)
        elif channel_type == "A":
            ax[idx].set_ylabel("Tilt (nrad)", fontsize=font)
        elif channel_type == "H":
            ax[idx].set_ylabel("Acceleration (nm/s²)", fontsize=font)

    for idx in [3, 4]:
        ax[idx].set_ylabel("Pressure (Pa)", fontsize=font)

    ax[Nrow-1].set_xlabel(f"Time ({time_unit}) from {config['tbeg']}", fontsize=font)

    if 'fmin' in config and 'fmax' in config:
        ax[0].set_title(f"{config['fmin']} - {config['fmax']} Hz")

    plt.tight_layout()
    
    if out:
        return fig