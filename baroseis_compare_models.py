# %% [markdown]
# # ROMY - Barometric Models

# %%
from calendar import c
import os
import obspy as obs
import matplotlib.pyplot as plt
import numpy as np

from src.baroseis import baroseis


stations = ["FUR", "ROMY", "DROMY"]
dates = ["20240312", "20240315", "20240316", "20240321", "20240423", "20240324", "20240829"]

# Frequency range
fmin = 5e-4
fmax = 1e-2


# select channel: A = tilt, J = rotation rate, H = acceleration
cha ="*A*"



for date in dates:

    for station in stations:

        # load config
        config = baroseis.load_from_yaml(f"./config/config_FFBI_{station}_{date}_file.yaml")

        path_to_figs = f"./figures/{date}/"

        # %% [markdown]
        # ### Load Spatial Pressure Gradients

        # %%
        from functions.read_sds import read_sds

        # %%
        # gradient = obs.read(f"./data/pressure_gradient_{date}.mseed")
        gradient = read_sds(f"./data/", "BW.PROMY.01.LDE", config['tbeg'], config['tend'])
        gradient += read_sds(f"./data/", "BW.PROMY.01.LDN", config['tbeg'], config['tend'])

        # process gradient
        gradient = gradient.detrend("linear")
        gradient = gradient.detrend("demean")
        gradient = gradient.taper(0.05, "cosine")
        gradient = gradient.filter("bandpass", freqmin=5e-4, freqmax=0.002, corners=4, zerophase=True)

        gradient = gradient.trim(config['tbeg'], config['tend'])
        gradient = gradient.detrend("demean")
        gradient = gradient.taper(0.05, "cosine")


        # %%
        # Initialize baroseis object
        bs = baroseis(conf=config)

        # Load data specified in config
        bs.load_data()

        # %%
        # bs.st.plot(equal_scale=False);

        # %%
        # band pass filer
        bs.filter_data(fmin=fmin, fmax=fmax)

        # detrend
        bs.st.detrend("demean")

        # taper edges
        bs.st.taper(0.1, "cosine")

        if station == "ROMY":
            # integrate rotation to tilt
            bs.integrate_data(method="cumtrapz") # method = "cumtrapz" or "spline"

        elif station == "FUR" or station == "DROMY":
            # convert acceleration to tilt
            for tr in bs.st:
                if tr.stats.channel[1] == "H":
                    tr.stats.channel = tr.stats.channel[0] + "A" + tr.stats.channel[-1]
                    if tr.stats.channel[-1] in ["N", "E"]:
                        tr.data = -tr.data/9.81
            
        # trim waveforms
        bs.st = bs.st.trim(bs.config['tbeg'], bs.config['tend'])

        # detrend waveforms
        bs.st.detrend("demean")

        # taper edges
        bs.st.taper(0.05, "cosine")

        # show new waveforms
        # bs.st.plot(equal_scale=False);


        # %%
        def model_tilt(seis_stream, pressure_data):
            """
            Simple model for predicting tilt/rotation from pressure data.
            
            Args:
                seis_stream: Stream with seismic data
                pressure_data: List of pressure arrays [P, H, DP, DH]
            
            Returns:
                Dictionary with predicted_data, coefficients, variance_reduction, residuals
            """
            
            # Get seismic data
            components = ['N', 'E', 'Z']
            seis_data = {}
            
            for comp in components:
                try:
                    tr = seis_stream.select(component=comp).copy()[0]
                    seis_data[comp] = tr.data
                except:
                    continue
            
            if not seis_data:
                raise ValueError("No seismic data found")
            
            # Ensure all data has same length
            data_length = len(pressure_data[0])
            for comp in seis_data:
                if len(seis_data[comp]) != data_length:
                    # Simple interpolation
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(0, 1, len(seis_data[comp]))
                    x_new = np.linspace(0, 1, data_length)
                    f = interp1d(x_old, seis_data[comp], kind='linear', fill_value='extrapolate')
                    seis_data[comp] = f(x_new)
            
            # Create design matrix
            A = np.column_stack(pressure_data)
            
            # Results
            results = {
                'predicted_data': {},
                'coefficients': {},
                'variance_reduction': {},
                'residuals': {}
            }
            
            # Process each component
            for comp, seis_comp_data in seis_data.items():
                # Least squares: A * x = b
                coefficients = np.linalg.lstsq(A, seis_comp_data, rcond=None)[0]
                predicted_data = A @ coefficients
                
                # Variance reduction
                original_var = np.var(seis_comp_data)
                residual_var = np.var(seis_comp_data - predicted_data)
                var_reduction = ((original_var - residual_var) / original_var) * 100
                
                # Store
                results['predicted_data'][comp] = predicted_data
                results['coefficients'][comp] = coefficients
                results['variance_reduction'][comp] = var_reduction
                results['residuals'][comp] = seis_comp_data - predicted_data
            
            return results

        # %% [markdown]
        # ### Model 1

        # %%
        seis_stream = bs.st.select(channel=cha).copy()
        model_data = [
            bs.st.select(channel="*DO").copy()[0].data,
            bs.st.select(channel="*DH").copy()[0].data,
        ]

        # Run model
        model1 = model_tilt(seis_stream, model_data)

        # Access results
        print("Variance reduction:")
        for comp in model1['predicted_data']:
            print(f"{comp}: {model1['variance_reduction'][comp]:.1f}%")


        # %% [markdown]
        # ### Model 2

        # %%
        seis_stream = bs.st.select(channel=cha).copy()
        pressure_data = [
            bs.st.select(channel="*DO").copy()[0].data,
            bs.st.select(channel="*DH").copy()[0].data,
            bs.st.select(channel="*DO").copy().differentiate()[0].data,
            bs.st.select(channel="*DH").copy().differentiate()[0].data
        ]

        # Run model
        model2 = model_tilt(seis_stream, pressure_data)

        # Access results
        print("Variance reduction:")
        for comp in model2['predicted_data']:
            print(f"{comp}: {model2['variance_reduction'][comp]:.1f}%")


        # %% [markdown]
        # ### Model 3

        # %%
        seis_stream = bs.st.select(channel=cha).copy()
        model_data = [
            bs.st.select(channel="*DO").copy()[0].data,
            # bs.st.select(channel="*DH").copy()[0].data,
            gradient.select(channel="*DE")[0].data,
            gradient.select(channel="*DN")[0].data,
        ]

        # Run model
        model3 = model_tilt(seis_stream, model_data)

        # Access results
        print("Variance reduction:")
        for comp in model3['predicted_data']:
            print(f"{comp}: {model3['variance_reduction'][comp]:.1f}%")

        # Get predicted data
        predicted_data = model3['predicted_data']
        coefficients = model3['coefficients']
        print(coefficients)

        # %% [markdown]
        # ### Model 4

        # %%
        # seis_stream = bs.st.select(channel=cha).copy()
        # model_data = [
        #     bs.st.select(channel="*DO").copy()[0].data,
        #     bs.st.select(channel="*DH").copy()[0].data,
        #     gradient.select(channel="*DE")[0].data,
        #     gradient.select(channel="*DN")[0].data,
        #     gradient.select(channel="*DE").copy().differentiate()[0].data,
        #     gradient.select(channel="*DN").copy().differentiate()[0].data,
        # ]

        # # Run model
        # model4 = model_tilt(seis_stream, model_data)

        # Access results
        # print("Variance reduction:")
        # for comp in model4['predicted_data']:
        #     print(f"{comp}: {model4['variance_reduction'][comp]:.1f}%")


        def plot_vr_comparison(model_list, figsize=(8, 5)):
            """
            Simple variance reduction comparison plot for a list of models.
            
            Args:
                model_list: List of dictionaries, each with 'name' and 'variance_reduction' keys
                figsize: Figure size tuple
            
            Example:
                models = [
                    {'name': 'Basic', 'variance_reduction': {'N': 45, 'E': 38, 'Z': 52}},
                    {'name': 'With Derivatives', 'variance_reduction': {'N': 67, 'E': 61, 'Z': 73}},
                    {'name': 'Advanced', 'variance_reduction': {'N': 72, 'E': 69, 'Z': 79}}
                ]
                plot_vr_comparison(models)
            """
            import matplotlib.pyplot as plt
            import numpy as np

            # Colors for different models
            colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
            
            components = ['N', 'E', 'Z']
            x = np.arange(len(components))
            width = 0.8 / len(model_list)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            for i, model in enumerate(model_list):
                vr_values = [model['variance_reduction'].get(comp, 0) for comp in components]
                ax.bar(x + i * width, vr_values, width, 
                    label=model['name'], color=colors[i % len(colors)], alpha=0.8)
                
            ax.set_ylabel('Variance Reduction (%)')
            ax.set_title('Variance Reduction Comparison')
            ax.set_xticks(x + width * (len(model_list) - 1) / 2)
            ax.set_xticklabels(f"{comp}-component" for comp in components)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            return fig

        # %%
        # Define your models
        models = [
            {'name': r'M1: P + H', 'variance_reduction': model1['variance_reduction']},
            {'name': r'M2: P + H + $\partial$P + $\partial$H', 'variance_reduction': model2['variance_reduction']},
            {'name': r'M3: P + G$_E$ + G$_N$', 'variance_reduction':model3['variance_reduction']},
            # {'name': r'M4: P + H + $\partial$P + $\partial$H + G$_E$ + G$_N$', 'variance_reduction':model4['variance_reduction']}
        ]
            # {'name': r'M3: P + H + G$_E$ + G$_N$', 'variance_reduction':model3['variance_reduction']},

        # Plot
        fig = plot_vr_comparison(models)

        fig.savefig(f"{path_to_figs}{date}_{station}_{cha.replace('*', '')}_model_vr_comparison.png", dpi=150, bbox_inches="tight")

        del fig
        # %%
        def plot_waveform_comparison(seis_stream, model_results,
                                time_unit='minutes', residual=False, figsize=(15, 10)):
            """
            Simple plot with 3 vertical subplots (Z, N, E) showing waveforms for each model.
            
            Args:
                seis_stream: Original seismic stream
                model_results: Dictionary with model names as keys and results as values
                time_unit: Time unit for x-axis
                residual: Boolean to plot residuals
                figsize: Figure size
            """
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get time scaling
            tscale_dict = {"hours": 1/3600, "days": 1/86400, "minutes": 1/60, "seconds": 1}
            tscale = tscale_dict.get(time_unit, 1/60)
            
            # Set units and scaling
            channel_type = seis_stream[0].stats.channel[1]
            if channel_type == 'J':
                ylabel = "Rotation Rate\n(nrad/s)"
                yscale = 1e9
            elif channel_type == 'A':
                ylabel = "Tilt (nrad)"
                yscale = 1e9
            elif channel_type == 'H':
                ylabel = "Acceleration\n(nm/s²)"
                yscale = 1e9
            else:
                ylabel = "Amplitude"
                yscale = 1.0

            components = ['Z', 'N', 'E']
            
            font = 14

            # Create 3 vertical subplots
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # Colors for different models
            colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
            
            # Plot each component
            for i, comp in enumerate(components):
                try:
                    # Get original data
                    orig_tr = seis_stream.select(channel=f"*{comp}").copy()[0]
                    times = orig_tr.times() * tscale
                    orig_data = orig_tr.data * yscale
                    
                    # Plot original data
                    axes[i].plot(times, orig_data, 'k-', linewidth=2, 
                                label=f'{comp}-Component', alpha=1, zorder=1)
                    max_orig = np.max(np.abs(orig_data))

                    # Plot each model
                    for j, (model_name, results) in enumerate(model_results.items()):
                        if comp in results['predicted_data']:
                            pred_data = results['predicted_data'][comp] * yscale
                            var_reduction = results['variance_reduction'][comp]
                            
                            color = colors[j % len(colors)]
                            if residual:
                                res_data = orig_data - pred_data
                                axes[i].plot(times, res_data, color=color, linewidth=1.5, zorder=2,
                                        label=f'M{j+1} (VR: {var_reduction:.1f}%)', alpha=0.9)
                                # find y absolute maximum for ylim
                                y_max = np.max([np.max(np.abs(res_data)), max_orig])*1.01
                                axes[i].set_ylim(-y_max, y_max)
                            else:
                                axes[i].plot(times, pred_data, color=color, linewidth=1.5, zorder=2,
                                        label=f'M{j+1} (VR: {var_reduction:.1f}%)', alpha=0.9)
                                # find y absolute maximum for ylim
                                y_max = np.max([np.max(np.abs(pred_data)), max_orig])*1.01
                                axes[i].set_ylim(-y_max, y_max)
                    # Format subplot
                    axes[i].set_ylabel(f"{ylabel}", fontsize=font)
                    axes[i].legend(fontsize=font-2, loc='lower left', ncol=5)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(labelsize=font-1)
            
                except Exception as e:
                    print(f"Error plotting {comp}: {e}")
                    axes[i].text(0.5, 0.5, f'Error loading {comp}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    continue
            
            # Set x-axis label
            axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=font)
            
            # Set overall title
            if residual:
                title = f"Residual Comparison"
            else:
                title = f"Model Comparison"
            fig.suptitle(title, fontsize=font+2, fontweight='bold')
            
            # Add model names as text outside the frame
            model_names = list(model_results.keys())
            model_text = "Models:  " + ",   ".join([f"{name}" for i, name in enumerate(model_names)])
            
            # Add text below the plot
            fig.text(0.5, 0.02, model_text, ha='center', va='bottom', fontsize=font-2, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            # Adjust layout to make room for the model text
            plt.subplots_adjust(bottom=0.1)
            return fig

        # %%
        # Example usage:
        model_results = {
            r'M1: P + H': model1,
            r'M2: P + H + $\partial$P + $\partial$H': model2,
            r'M3: P + G$_E$ + G$_N$': model3,
            # r'M4: P + H + $\partial$P + $\partial$H + G$_E$ + G$_N$': model4
        }
            # r'M3: P + H + G$_E$ + G$_N$': model3,

        fig = plot_waveform_comparison(seis_stream, model_results)

        fig.savefig(f"{path_to_figs}{date}_{station}_{cha.replace('*', '')}_model_waveform_comparison.png", dpi=150, bbox_inches="tight")

        del fig

        fig = plot_waveform_comparison(seis_stream, model_results, residual=True)

        fig.savefig(f"{path_to_figs}{date}_{station}_{cha.replace('*', '')}_model_waveform_residual_comparison.png", dpi=150, bbox_inches="tight")

        del fig
        
        # %%
        def plot_waveforms_with_gradient(st_plot, config, gradient_stream=None, fmin=None, fmax=None,
                                        time_unit="hours", channel_type="A", out=False):
            """
            Plot waveforms including tilt, pressure, hilbert, and gradient components.
            Shows cross-correlation values with tilt N and E components.
            
            Args:
                st_plot: Stream containing tilt (N, E, Z), pressure (DO), hilbert (DH), and optionally gradient (DE, DN)
                config: Configuration dictionary with tbeg, tend, fmin, fmax
                gradient_stream: Optional separate stream for gradient data (if not in st_plot)
                time_unit: Time unit for x-axis ('hours', 'days', 'minutes', 'seconds')
                channel_type: Channel type ('J' for rotation rate, 'A' for tilt, 'H' for acceleration)
                out: Return figure handle if True
                fmin: Minimum frequency for title
                fmax: Maximum frequency for title
            Returns:
                matplotlib.figure.Figure if out=True
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from obspy.signal.cross_correlation import correlate
            
            Nrow = 7  # Z, N, E tilt, Pressure, Hilbert, Gradient N, Gradient E
            yscale = 1e9
            font = 12
            
            # Set time scaling
            tscale_dict = {
                "hours": 1/3600,
                "days": 1/86400,
                "minutes": 1/60,
                "seconds": 1
            }
            tscale = tscale_dict.get(time_unit, 1/3600)
            
            fig, ax = plt.subplots(Nrow, 1, figsize=(14, 10))
            
            # Get tilt N and E data for cross-correlation
            try:
                tr_tilt_n = st_plot.select(channel=f"*{channel_type}N").copy()[0]
                tr_tilt_e = st_plot.select(channel=f"*{channel_type}E").copy()[0]
                tilt_n_data = tr_tilt_n.data
                tilt_e_data = tr_tilt_e.data
                sampling_rate = tr_tilt_n.stats.sampling_rate
            except:
                print("Warning: Could not find tilt N/E components for cross-correlation")
                tilt_n_data = None
                tilt_e_data = None
                sampling_rate = 1.0
            
            # Helper function to calculate zero-lag cross-correlation coefficient
            def calc_cc0(data1, data2):
                """Calculate zero-lag cross-correlation coefficient (normalized Pearson correlation)."""
                if data1 is None or data2 is None:
                    return None
                min_len = min(len(data1), len(data2))
                if min_len == 0:
                    return None
                d1 = data1[:min_len]
                d2 = data2[:min_len]
                
                # Use numpy corrcoef for normalized correlation coefficient
                corr_matrix = np.corrcoef(d1, d2)
                if np.isnan(corr_matrix[0, 1]):
                    return 0.0
                return corr_matrix[0, 1]
            
            # Plot tilt components (Z, N, E)
            for comp, color, idx in zip(['Z', 'N', 'E'], ['tab:blue', 'tab:orange', 'tab:red'], range(3)):
                try:
                    tr = st_plot.select(channel=f"*{channel_type}{comp}").copy()[0]
                    times = tr.times(reftime=config['tbeg'])*tscale
                    data = tr.data*yscale
                    
                    ax[idx].plot(times, data, label=f"{comp}-Component", color=color, linewidth=1.5)
                    ax[idx].fill_between(times, 0, data,
                                       where=data>0, interpolate=True,
                                       color=color, alpha=0.3)
                    ax[idx].set_ylim(-np.max(np.abs(data))*1.1, np.max(np.abs(data))*1.1)
                except Exception as e:
                    print(f"Could not plot tilt component {comp}: {str(e)}")
            
            # Plot pressure
            try:
                tr_p = st_plot.select(channel="*DO").copy()[0]
                times = tr_p.times(reftime=config['tbeg'])*tscale
                data = tr_p.data
                
                # Calculate cross-correlations
                cc_n = calc_cc0(data, tilt_n_data)
                cc_e = calc_cc0(data, tilt_e_data)
                
                ax[3].plot(times, data, label="Pressure", color='k', linewidth=1.5)
                ax[3].fill_between(times, 0, data,
                                  where=data>0, interpolate=True,
                                  color='k', alpha=0.3)
                ax[3].set_ylim(-np.max(np.abs(data)), np.max(np.abs(data)))

                # Add cross-correlation text
                if cc_n is not None and cc_e is not None:
                    text_str = f"CC$_N$={cc_n:.1f}, CC$_E$={cc_e:.1f}"
                    ax[3].text(0.02, 0.95, text_str, transform=ax[3].transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
            except Exception as e:
                print(f"Could not plot pressure data: {str(e)}")
            
            # Plot Hilbert transform
            try:
                tr_h = st_plot.select(channel="*DH").copy()[0]
                times = tr_h.times(reftime=config['tbeg'])*tscale
                data = tr_h.data
                
                # Calculate cross-correlations
                cc_n = calc_cc0(data, tilt_n_data)
                cc_e = calc_cc0(data, tilt_e_data)
                
                ax[4].plot(times, data, label="Hilbert(Pressure)", color='darkgrey', linewidth=1.5)
                ax[4].fill_between(times, 0, data,
                                  where=data>0, interpolate=True,
                                  color='darkgrey', alpha=0.3)
                ax[4].set_ylim(-np.max(np.abs(data))*1.1, np.max(np.abs(data))*1.1)

                # Add cross-correlation text
                if cc_n is not None and cc_e is not None:
                    text_str = f"CC$_N$={cc_n:.1f}, CC$_E$={cc_e:.1f}"
                    ax[4].text(0.02, 0.95, text_str, transform=ax[4].transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
            except Exception as e:
                print(f"Could not plot Hilbert transform: {str(e)}")
            
            # Plot gradient components (N, E)
            gradient_st = gradient_stream if gradient_stream is not None else st_plot
            
            # Gradient North
            try:
                tr_gn = gradient_st.select(channel="*DN").copy()[0]
                times = tr_gn.times(reftime=config['tbeg'])*tscale
                data = tr_gn.data * 1e3  # Convert to Pa/km
                
                # Calculate cross-correlations
                cc_n = calc_cc0(data, tilt_n_data)
                cc_e = calc_cc0(data, tilt_e_data)
                
                ax[5].plot(times, data, label="Gradient North", color='tab:pink', linewidth=1.5)
                ax[5].fill_between(times, 0, data,
                                  where=data>0, interpolate=True,
                                  color='tab:pink', alpha=0.3)
                ax[5].set_ylim(-np.max(np.abs(data))*1.1, np.max(np.abs(data))*1.1)

                # Add cross-correlation text
                if cc_n is not None and cc_e is not None:
                    text_str = f"CC$_N$={cc_n:.1f}, CC$_E$={cc_e:.1f}"
                    ax[5].text(0.02, 0.95, text_str, transform=ax[5].transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='w', alpha=0.5))
            except Exception as e:
                print(f"Could not plot gradient North: {str(e)}")
            
            # Gradient East
            try:
                tr_ge = gradient_st.select(channel="*DE").copy()[0]
                times = tr_ge.times(reftime=config['tbeg'])*tscale
                data = tr_ge.data * 1e3  # Convert to Pa/km
                
                # Calculate cross-correlations
                cc_n = calc_cc0(data, tilt_n_data)
                cc_e = calc_cc0(data, tilt_e_data)
                
                ax[6].plot(times, data, label="Gradient East", color='tab:purple', linewidth=1.5)
                ax[6].fill_between(times, 0, data,
                                  where=data>0, interpolate=True,
                                  color='tab:purple', alpha=0.3)
                ax[6].set_ylim(-np.max(np.abs(data))*1.1, np.max(np.abs(data))*1.1)
                
                # Add cross-correlation text
                if cc_n is not None and cc_e is not None:
                    text_str = f"CC$_N$={cc_n:.1f}, CC$_E$={cc_e:.1f}"
                    ax[6].text(0.02, 0.95, text_str, transform=ax[6].transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            except Exception as e:
                print(f"Could not plot gradient East: {str(e)}")
            
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
            
            ax[3].set_ylabel("P (Pa)", fontsize=font)
            ax[4].set_ylabel("H(P) (Pa)", fontsize=font)
            ax[5].set_ylabel("Gradient\nNorth (Pa/km)", fontsize=font)
            ax[6].set_ylabel("Gradient\nEast (Pa/km)", fontsize=font)
            
            ax[Nrow-1].set_xlabel(f"Time ({time_unit}) from {config['tbeg'].strftime('%Y-%m-%d %H:%M:%S')} UTC", fontsize=font)
            
            for n in range(Nrow):
                ax[n].set_xlim(left=0)

            if fmin is not None and fmax is not None:
                ax[0].set_title(f"f = {fmin*1e3:.1f} - {fmax*1e3:.1f} mHz", fontsize=font)
            
            plt.tight_layout()
            
            if out:
                return fig
            else:
                plt.show()
        
        # %%
        # Plot waveforms with gradient
        fig = plot_waveforms_with_gradient(
            bs.st.copy(),
            config,
            gradient_stream=gradient, 
            time_unit="minutes",
            channel_type=cha[1],
            out=True,
            fmin=fmin,
            fmax=fmax
        )

        fig.savefig(f"{path_to_figs}{date}_{station}_{cha.replace('*', '')}_waveforms.png", dpi=150, bbox_inches="tight")
        del fig
        
        # %%
        def plot_spectra_comparison(seis_stream, model_results,
                                method='welch', fmin=0.0005, fmax=0.03,
                                log_scale=True, db_scale=False, residual=False, 
                                smooth_octave=False, octave_fraction=1/3, 
                                smooth_method='median', figsize=(15, 10)):
            """
            Plot spectra comparison for 3 components (Z, N, E) showing spectra for each model.
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.signal import welch
            from scipy.fft import fft, fftfreq
            from scipy.signal import windows

            def smooth_octave_bands(freq, spectrum, octave_fraction=1/3, method='median'):
                """Smooth spectrum in fractional octave bands."""
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

            # Set units and scaling
            channel_type = seis_stream[0].stats.channel[1]
            if channel_type == 'J':
                ylabel = "Rotation Rate\n(nrad/s)"
                yscale = 1e9
            elif channel_type == 'A':
                ylabel = "Tilt (nrad)"
                yscale = 1e9
            elif channel_type == 'H':
                ylabel = "Acceleration\n(nm/s²)"
                yscale = 1e9
            else:
                ylabel = "Amplitude"
                yscale = 1.0

            components = ['Z', 'N', 'E']
            font = 14

            # Create 3 vertical subplots
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # Colors for different models
            colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
            
            # Plot each component
            for i, comp in enumerate(components):
                try:
                    # Get original data
                    orig_tr = seis_stream.select(channel=f"*{comp}").copy()[0]
                    
                    # Compute original spectrum
                    if method.lower() == 'fft':
                        n = len(orig_tr.data)
                        win = windows.hann(n)
                        spec_orig = fft(orig_tr.data * win)
                        freq = fftfreq(n, d=orig_tr.stats.delta)
                        pos_freq = freq[0:n//2]
                        mag_orig = np.abs(spec_orig[0:n//2]) * 2.0/n * yscale
                    else:  # welch
                        nperseg = int(orig_tr.stats.sampling_rate * 3600)  # 1-hour segments
                        noverlap = nperseg // 2
                        freq, psd_orig = welch(orig_tr.data, fs=orig_tr.stats.sampling_rate,
                                            window='hann', nperseg=nperseg, noverlap=noverlap)
                        mag_orig = np.sqrt(psd_orig) * yscale
                        pos_freq = freq
                    
                    # Apply octave band smoothing if requested
                    if smooth_octave:
                        pos_freq, mag_orig = smooth_octave_bands(pos_freq, mag_orig, 
                                                                octave_fraction, smooth_method)
                    
                    # Convert to dB if requested
                    if db_scale:
                        mag_orig = 20 * np.log10(mag_orig)
                        ylabel_comp = ylabel + " (dB)"
                    else:
                        ylabel_comp = ylabel
                    
                    # Apply frequency limits
                    mask = (pos_freq >= fmin) & (pos_freq <= fmax)
                    pos_freq_plot = pos_freq[mask]
                    mag_orig_plot = mag_orig[mask]
                    
                    # Plot original spectrum
                    axes[i].plot(pos_freq_plot, mag_orig_plot, 'k-', linewidth=2, 
                                label=f'{comp}-Component', alpha=1, zorder=1)

                    # Plot each model
                    for j, (model_name, results) in enumerate(model_results.items()):
                        if comp in results['predicted_data']:
                            residual_tr = seis_stream.select(channel=f"*{comp}").copy()[0]
                            residual_tr.data = seis_stream.select(channel=f"*{comp}").copy()[0].data - results['predicted_data'][comp]
                            var_reduction = results['variance_reduction'][comp]
                            
                            # Compute predicted spectrum
                            if method.lower() == 'fft':
                                n = len(residual_tr.data)
                                win = windows.hann(n)
                                spec_pred = fft(residual_tr.data * win)
                                mag_pred = np.abs(spec_pred[0:n//2]) * 2.0/n * yscale
                                freq_pred = fftfreq(n, d=residual_tr.stats.delta)[0:n//2]
                            else:  # welch
                                nperseg = int(residual_tr.stats.sampling_rate * 3600)
                                noverlap = nperseg // 2
                                freq_pred, psd_pred = welch(residual_tr.data, fs=residual_tr.stats.sampling_rate,
                                                        window='hann', nperseg=nperseg, noverlap=noverlap)
                                mag_pred = np.sqrt(psd_pred) * yscale
                            
                            # Apply octave band smoothing if requested
                            if smooth_octave:
                                freq_pred, mag_pred = smooth_octave_bands(freq_pred, mag_pred, 
                                                                        octave_fraction, smooth_method)
                            
                            # Convert to dB if requested
                            if db_scale:
                                mag_pred = 20 * np.log10(mag_pred)
                            
                            # Apply frequency limits
                            mask_pred = (freq_pred >= fmin) & (freq_pred <= fmax)
                            pos_freq_pred_plot = freq_pred[mask_pred]
                            mag_pred_plot = mag_pred[mask_pred]
                            
                            color = colors[j % len(colors)]
                            if residual:
                                # For residuals, we need to interpolate to match frequencies
                                from scipy.interpolate import interp1d
                                f_interp = interp1d(pos_freq_plot, mag_orig_plot, kind='linear', 
                                                bounds_error=False, fill_value=0)
                                mag_orig_interp = f_interp(pos_freq_pred_plot)
                                res_spectrum = mag_orig_interp - mag_pred_plot
                                axes[i].plot(pos_freq_pred_plot, res_spectrum, color=color, linewidth=1.5, zorder=2,
                                        label=f'M{j+1} (VR: {var_reduction:.1f}%)', alpha=0.9)
                            else:
                                axes[i].plot(pos_freq_pred_plot, mag_pred_plot, color=color, linewidth=1.5, zorder=2,
                                        label=f'M{j+1} (VR: {var_reduction:.1f}%)', alpha=0.9)
                    
                    # Format subplot
                    if db_scale:
                        axes[i].set_ylabel(r"ASD (dB wrt. nrad/$\sqrt{Hz}$)", fontsize=font)
                    else:
                        axes[i].set_ylabel(r"ASD nrad/$\sqrt{Hz}$", fontsize=font)
                    axes[i].legend(loc='lower left', fontsize=font-2, ncol=5)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(labelsize=font-1)
                    axes[i].set_xlim(fmin, fmax)
                    
                    # Set scales
                    if log_scale:
                        axes[i].set_xscale('log')
                        if not db_scale:  # Only use log scale for y-axis if not in dB
                            axes[i].set_yscale('log')
            
                except Exception as e:
                    print(f"Error plotting {comp}: {e}")
                    axes[i].text(0.5, 0.5, f'Error loading {comp}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    continue
            
            # Set x-axis label
            axes[-1].set_xlabel("Frequency (Hz)", fontsize=font)
            
            # Set overall title
            if residual:
                title = f"Spectra Residual Comparison"
            else:
                title = f"Spectra Model Comparison"
            
            fig.suptitle(title, fontsize=font+2, fontweight='bold')
            
            # Add model names as text outside the frame
            model_names = list(model_results.keys())
            model_text = "Models:  " + ",   ".join([f"{name}" for i, name in enumerate(model_names)])
            
            # Add text below the plot
            fig.text(0.5, 0.02, model_text, ha='center', va='bottom', fontsize=font-2, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            # Adjust layout to make room for the model text
            plt.subplots_adjust(bottom=0.1)

            return fig

        # %%
        # ASD scale
        fig = plot_spectra_comparison(seis_stream, model_results, method='fft', db_scale=False, fmin=fmin, fmax=fmax)

        fig.savefig(f"{path_to_figs}{date}_{station}_{cha.replace('*', '')}_model_spectra_comparison.png", dpi=150, bbox_inches="tight")

        del fig

        # dB scale
        fig = plot_spectra_comparison(seis_stream, model_results, method='fft', db_scale=True, fmin=fmin, fmax=fmax)

        fig.savefig(f"{path_to_figs}{date}_{station}_{cha.replace('*', '')}_model_spectra_comparison_db.png", dpi=150, bbox_inches="tight")

        del fig
       