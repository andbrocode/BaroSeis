"""
Module for plotting residuals between observed and predicted data with derivatives.
"""

from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, UTCDateTime

def plot_residuals_derivatives(st: Stream, config: Dict[str, Any], time_unit: str = "minutes", 
                              channel_type: str = "J", out: bool = False) -> Optional[plt.Figure]:
    """
    Plot residuals between observed and predicted data using model_tilt_from_pressure output.
    
    This function creates a waveform plot similar to plot_residuals but uses the model
    output from model_tilt_from_pressure which includes derivatives of pressure and
    Hilbert transform.
    
    Args:
        st: Stream containing the data
        config: Configuration dictionary
        time_unit: Time unit for x-axis ('minutes', 'hours', 'days', 'seconds')
        channel_type: Type of channel to plot:
                     'J' for rotation rate
                     'A' for tilt
                     'H' for acceleration
        out: If True, return figure object
    """
    # Set units based on channel type
    if channel_type == 'J':
        ylabel = "Rotation Rate\n(nrad/s)"
        yscale = 1e9
        coef_unit = "nrad/s/hPa"
    elif channel_type == 'A':
        ylabel = "Tilt (nrad)"
        yscale = 1e9
        coef_unit = "nrad/hPa"
    elif channel_type == 'H':
        ylabel = "Acceleration\n(nm/s²)"
        yscale = 1e9
        coef_unit = "nm/s²/hPa"
    else:
        ylabel = "Amplitude"
        yscale = 1.0
        coef_unit = "units/hPa"
    
    Nrow, Ncol = 6, 1
    font = 13
    
    # Set time scaling
    tscale_dict = {
        "hours": 1/3600,
        "days": 1/86400,
        "minutes": 1/60,
        "seconds": 1
    }
    tscale = tscale_dict.get(time_unit, 1/60)  # Default to minutes

    # Create figure with shared x-axis
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 10), sharex=True)
    
    # Plot each component (N, E, Z) with its residual
    components = ['N', 'E', 'Z']
    for i, comp in enumerate(components):
        idx = i * 2  # Index for main plot

        try:
            # Get rotation/tilt data
            tr_rot = st.select(channel=f"*{channel_type}{comp}").copy()[0]
            times = tr_rot.times(reftime=config['tbeg'])*tscale
            rot_data = tr_rot.data * yscale

            # Define label before the if block
            # label = f"{'Tilt' if channel_type == 'A' else 'Rotation'}-{comp}"
            label = f"{comp}-Component"

            # Get predicted data if available - look for PP location with same channel
            tr_pred = (st.select(location="PP", channel=f"*{channel_type}{comp}").copy()[0]
                      if st.select(location="PP", channel=f"*{channel_type}{comp}").copy()
                      else None)
            
            if tr_pred is not None:
                pred_data = tr_pred.data * yscale
                residual = rot_data - pred_data
                
                # Calculate variance reduction
                var_red = ((np.var(rot_data) - np.var(residual)) / np.var(rot_data) * 100)
                
                # Get max amplitude for y-axis scaling
                y_max = max([max(abs(rot_data)), max(abs(pred_data))])
                
                # Get coefficients if available from model_tilt_from_pressure
                coef_text = ""
                if 'p_coefficient' in config and 'h_coefficient' in config:
                    if comp in config['p_coefficient'] and comp in config['h_coefficient']:
                        p_coef = config['p_coefficient'][comp] * 1e9 * 1e2  # Convert to nano units and hPa
                        h_coef = config['h_coefficient'][comp] * 1e9 * 1e2  # Convert to nano units and hPa
                        coef_text = f"P: {p_coef:.0f} {coef_unit}\nH: {h_coef:.0f} {coef_unit}"
                        
                        # Add derivative coefficients if available
                        if 'dp_coefficient' in config and 'dh_coefficient' in config:
                            if comp in config['dp_coefficient'] and comp in config['dh_coefficient']:
                                dp_coef = config['dp_coefficient'][comp] * 1e9 * 1e2
                                dh_coef = config['dh_coefficient'][comp] * 1e9 * 1e2
                                coef_text += f"\n$\delta_t$P: {dp_coef:.0f} s{coef_unit}\n$\delta_t$H: {dh_coef:.0f} s{coef_unit}"
                
                # Plot original and predicted data
                ax[idx].plot(times, rot_data, label=label, color='black')
                ax[idx].plot(times, pred_data, label=f"Model", color='red')
                ax[idx].set_ylim(-y_max, y_max)
                ax[idx].set_ylabel(ylabel, fontsize=font)
                ax[idx].legend(loc='upper right', fontsize=font-1)
                
                # Add coefficient text
                if coef_text:
                    ax[idx].text(
                        0.02, 0.98, coef_text, 
                        backgroundcolor='white', 
                        bbox=dict(alpha=0.7, facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                        transform=ax[idx].transAxes,
                        va='top', ha='left', fontsize=font-1
                    )
                
                # Plot residual
                ax[idx+1].plot(times, residual, color="grey", 
                              label=f"VR={var_red:.1f}%")
                ax[idx+1].set_ylim(-y_max, y_max)
                ax[idx+1].set_ylabel(ylabel, fontsize=font)
                ax[idx+1].legend(loc='upper right', fontsize=font-1)
            
            else:
                # Get max amplitude for y-axis scaling
                y_max = max(abs(rot_data))
                
                # Plot only original data
                ax[idx].plot(times, rot_data, label=label, color='black')
                ax[idx].set_ylim(-y_max, y_max)
                ax[idx].set_ylabel(ylabel, fontsize=font)
                ax[idx].legend(loc='upper right', fontsize=font-1)
                
                ax[idx+1].text(0.5, 0.5, 'No prediction available', 
                              ha='center', va='center',
                              transform=ax[idx+1].transAxes)
            
        except Exception as e:
            print(f"Could not plot component {comp}: {str(e)}")
            continue
    
    # Format axes
    for i in range(Nrow):
        ax[i].spines[['right', 'top']].set_visible(False)
        if i < Nrow-1:
            ax[i].spines[['bottom']].set_visible(False)
        ax[i].grid(True, alpha=0.3)
        # set size of ticks font
        ax[i].tick_params(axis='both', which='major', labelsize=font-2)
    
    # Set title with time range and frequency band
    title = (f"{config['tbeg'].date} {str(config['tbeg'].time).split('.')[0]} - "
            f"{str(config['tend'].time).split('.')[0]} UTC")
    if 'fmin' in config and 'fmax' in config:
        title += f" | f = {config['fmin']*1e3:.1f} - {config['fmax']*1e3:.1f} mHz"
    ax[0].set_title(title, fontsize=font)
    
    # Set x-axis label
    ax[Nrow-1].set_xlabel(f"Time ({time_unit})", fontsize=font)
    
    plt.tight_layout()
    
    if out:
        return fig
