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

    # Extract unique components from the stream
    components = []
    for tr in st:
        if tr.stats.channel[1] == channel_type:
            comp = tr.stats.channel[2]
            if comp not in components:
                components.append(comp)
    
    # Check if we have any components
    if len(components) == 0:
        raise ValueError(f"No components found for channel type '{channel_type}' in stream")

    Nrow, Ncol = len(components)+2, 1
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
    if Nrow == 1:
        ax = [ax]  # Make it iterable if only one subplot

    # Colors for components (cycle if more than 3)
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown']
    
    # Plot rotation components
    for idx, (comp, color) in enumerate(zip(components, colors[:len(components)])):
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

    try:
        # Plot original pressure data
        tr_p = st.select(channel="*DO").copy()[0]  # Pressure channel
        times = tr_p.times(reftime=config['tbeg'])*tscale
        data = tr_p.data
        
        ax[Nrow-2].plot(times, data, label="Pressure", color='k')
        ax[Nrow-2].fill_between(times, 0, data,
                          where=data>0, interpolate=True,
                          color='k', alpha=0.5)
    except Exception as e:
        print(f"Could not plot pressure data: {str(e)}")

    try:
        # Plot Hilbert transform
        tr_h = st.select(channel="*DH").copy()[0]  # Hilbert transform channel
        times = tr_h.times(reftime=config['tbeg'])*tscale
        data = tr_h.data
        
        ax[Nrow-1].plot(times, data, label="Hilbert(Pressure)", color='darkgrey')
        ax[Nrow-1].fill_between(times, 0, data,
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
    for idx in range(len(components)):
        if channel_type == "J":
            ax[idx].set_ylabel("Rotation Rate\n(nrad/s)", fontsize=font)
        elif channel_type == "A":
            ax[idx].set_ylabel("Tilt (nrad)", fontsize=font)
        elif channel_type == "H":
            ax[idx].set_ylabel("Acceleration (nm/s²)", fontsize=font)

    for idx in [Nrow-2, Nrow-1]:
        ax[idx].set_ylabel("Pressure (Pa)", fontsize=font)

    ax[Nrow-1].set_xlabel(f"Time ({time_unit}) from {config['tbeg']}", fontsize=font)

    if 'fmin' in config and 'fmax' in config:
        ax[0].set_title(f"{config['fmin']} - {config['fmax']} Hz")

    plt.tight_layout()
    
    if out:
        return fig