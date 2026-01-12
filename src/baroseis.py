#!/usr/bin/python
#
# Class for analyzing barometer and rotation/seismometer data
#
# by Andreas Brotzer @2025
#

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from obspy import Stream, UTCDateTime, read_inventory
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.rotate import rotate_ne_rt
from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from numpy import ndarray
from obspy import Stream, Trace, UTCDateTime, Inventory
from pathlib import Path
from obspy.geodetics.base import gps2dist_azimuth
from obspy.clients.filesystem.sds import Client as SDSClient

from .utils.decimator import Decimator
from .plots.plot_coherence import plot_coherence
from .plots.plot_cross_correlations import plot_cross_correlations
from .plots.plot_cwt import plot_cwt
from .plots.plot_waveforms import plot_waveforms
from .plots.plot_residuals import plot_residuals
from .plots.plot_residuals_derivatives import plot_residuals_derivatives
from .plots.plot_scatter_correlations import plot_scatter_correlations
# from .plots.plot_spectra import compare_spectra


# =============================================================================
# Main Class Definition
# =============================================================================

class baroseis:
    """
    A class for analyzing and processing barometer and rotation/seismometer data.
    
    This class provides functionality for:
    - Loading and preprocessing barometer and seismometer/rotation data
    - Computing correlations between pressure and rotation/tilt signals
    - Predicting tilt/rotation from pressure data using various regression methods
    - Analyzing data in frequency domain (FFT, coherence, wavelet transforms)
    - Visualizing results through various plotting functions
    
    The class supports both SDS (Seismic Data Structure) and FDSN web service data sources,
    and handles various data formats and channel types.
    
    Attributes:
        st_baro (Stream): Barometer data stream
        st_seis (Stream): Seismometer/rotation sensor data stream
        st (Stream): Combined stream containing all data
        config (Dict): Configuration dictionary containing analysis parameters
        baro_inv (Inventory): Barometer station metadata
        seis_inv (Inventory): Seismometer station metadata
    
    Example:
        >>> # Initialize with basic configuration
        >>> config = {
        ...     'baro_seed': 'BW.ROMY..BDX',
        ...     'seis_seeds': ['BW.ROMY..BJZ', 'BW.ROMY..BJN', 'BW.ROMY..BJE'],
        ...     'tbeg': '2024-03-15T15:00:00',
        ...     'tend': '2024-03-15T18:00:00',
        ...     'path_to_baro_data': '/path/to/baro/sds',
        ...     'path_to_seis_data': '/path/to/seis/sds'
        ... }
        >>> bs = baroseis(config)
        >>> 
        >>> # Load and process data
        >>> bs.load_data()
        >>> bs.filter_data(fmin=0.0001, fmax=0.01)
        >>> 
        >>> # Predict tilt from pressure
        >>> bs.predict_tilt_from_pressure(method='regression', reg_type='theilsen')
        >>> 
        >>> # Plot results
        >>> bs.plot_residuals()
    
    Notes:
        - The class expects consistent sampling rates between barometer and seismometer data
        - Default configuration values are set for ROMY station parameters
        - Supports various regression methods for tilt prediction: least squares, RANSAC, TheilSen
        - Provides extensive plotting capabilities for time series and spectral analysis
    """

    # Class attributes with type hints
    st_baro: Stream
    st_seis: Stream
    st: Stream
    st0: Stream
    config: Dict[str, Any]
    baro_inv: Optional[Inventory]
    seis_inv: Optional[Inventory]
    p_coefficient: Dict[str, float]
    h_coefficient: Dict[str, float]
    st_baro_hilbert: Stream

    def __init__(self, conf: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the baroseis object.

        This method initializes a new baroseis instance with the provided configuration.
        If no configuration is provided, default values are used. The method sets up
        empty streams and initializes all required attributes.

        Args:
            conf: Configuration dictionary with analysis parameters. See class docstring
                 for available configuration options and their default values.

        Raises:
            TypeError: If conf is provided but is not a dictionary
            ValueError: If conf contains invalid parameter values

        Example:
            >>> # Initialize with default configuration
            >>> bs = baroseis()
            >>> # Initialize with custom configuration
            >>> bs = baroseis({
            ...     'sampling_rate': 1.0,
            ...     'fmin': 0.0001,
            ...     'fmax': 0.01
            ... })

        Notes:
            - All configuration parameters have sensible defaults
            - Time strings are automatically converted to UTCDateTime objects
            - Empty streams are initialized but not populated until load_data() is called
        """
        # Define required configuration keys and their default values
        default_config = {
            # Time parameters
            'tbeg': None,
            'tend': None,
            
            # Station parameters
            'station_latitude': 48.162941,  # ROMY latitude
            'station_longitude': 11.275501, # ROMY longitude
            'station_elevation': 560.0,     # ROMY elevation
            
            # time window parameters
            'time_buffer': 3600.0,  # Time buffer in seconds

            # Processing parameters
            'sampling_rate': 1.0,  # Sampling rate in Hz
            'fmin': 0.0001,  # Minimum frequency in Hz
            'fmax': 0.01,    # Maximum frequency in Hz
            'win_length_sec': 300.0,  # Window length for analysis in seconds
            'overlap': 0.5,  # Window overlap fraction
            'step': 1,      # Step size for analysis
            
            # Cross-correlation parameters
            'cc_threshold': 0.7,  # Minimum correlation coefficient
            
            # Data archive path
            'path_to_baro_data': None,
            'path_to_seis_data': None,

            # Inventory files
            'baro_inventory': None,
            'seis_inventory': None,
            'metadata_correction': True,

            # data source parameters
            'data_source': 'sds',  # data source ("sds" or "fdsn")

            # FDSN parameters
            'fdsn_server': None,  # FDSN server

            # Channel configurations
            'baro_seed': None,  # e.g. "BW.ROMY..BDX"
            'seis_seeds': [],   # e.g. ["BW.ROMY..BJZ", "BW.ROMY..BJN", "BW.ROMY..BJE"]
                               # or ["BW.ROMY..BHZ", "BW.ROMY..BHN", "BW.ROMY..BHE"]
            
            # Other parameters
            'verbose': False,
            
            # Output parameters
            'path_to_figures': './figures',
            'out_seed': None,

            # remove response parameters
            'remove_baro_response': True,
            'remove_seis_sensitivity': True,
            'remove_seis_response': True,

        }
        
        # Initialize config with default values
        self.config = default_config.copy()
        
        # Update with provided configuration if any
        if conf is not None:
            self.config.update(conf)
        
        # Initialize empty streams and inventories
        self.st_baro = Stream()
        self.st_seis = Stream()
        self.baro_inv = None
        self.seis_inv = None

        self.sampling_rate = self.config['sampling_rate']

        # Convert time strings to UTCDateTime if provided
        if 'tbeg' in self.config.keys() and self.config['tbeg'] is not None:
            self.config['tbeg'] = UTCDateTime(self.config['tbeg'])
        if 'tend' in self.config.keys() and self.config['tend'] is not None:
            self.config['tend'] = UTCDateTime(self.config['tend'])


    # =========================================================================
    # Private Utility Methods
    # =========================================================================

    def load_data(self, 
                tbeg: Optional[str] = None, 
                tend: Optional[str] = None,
                ) -> None:
        """
        Load barometer and seismometer/rotation data from specified data source.
        
        This method loads data based on the configuration parameters and performs initial
        preprocessing steps. It supports multiple data sources (SDS, FDSN, file) and
        handles various data formats and channel types.

        The method performs the following steps:
        1. Validates and processes time parameters
        2. Loads data from the configured source (SDS/FDSN/file)
        3. Applies metadata corrections if configured
        4. Merges data streams to handle gaps
        5. Performs quality checks (gaps, sampling rates)
        6. Adds Hilbert transform for analysis
        7. Detrends and resamples to common sampling rate

        Args:
            tbeg: Start time in any format accepted by UTCDateTime.
                 If provided, overrides config['tbeg'].
            tend: End time in any format accepted by UTCDateTime.
                 If provided, overrides config['tend'].

        Raises:
            ValueError: If required configuration parameters are missing
            ValueError: If data source is invalid or not specified
            ValueError: If time window is invalid
            IOError: If data files cannot be accessed
            TypeError: If time parameters are in invalid format

        Example:
            >>> bs = baroseis(config)
            >>> # Load data for specific time window
            >>> bs.load_data(
            ...     tbeg='2024-03-15T15:00:00',
            ...     tend='2024-03-15T18:00:00'
            ... )
            >>> print(bs.st)  # Show loaded streams

        Notes:
            - The method requires proper configuration of data sources
            - Data quality checks are performed automatically
            - Metadata corrections are applied based on configuration
            - Runtime information is printed if verbose mode is enabled
        """
        import timeit
        
        # convert tbeg and tend to UTCDateTime if provided
        if tbeg is not None:
            self.config['tbeg'] = UTCDateTime(tbeg)
        if tend is not None:
            self.config['tend'] = UTCDateTime(tend)

        # Start timer
        start_timer = timeit.default_timer()
        
        # Check required parameters
        if not self.config['baro_seed']:
            raise ValueError("baro_seed not specified in config")
        if not self.config['seis_seeds']:
            raise ValueError("seis_seeds not specified in config")
        if not self.config['tbeg'] or not self.config['tend']:
            raise ValueError("Time window (tbeg, tend) not specified in config or as arguments")

        # add time buffer
        self.config['t1'] = self.config['tbeg'] - self.config['time_buffer']
        self.config['t2'] = self.config['tend'] + self.config['time_buffer']

        # Load data based on source type
        if self.config['data_source'].lower() == "sds":
            if not self.config['path_to_baro_data'] or not self.config['path_to_seis_data']:
                raise ValueError("path_to_baro_data or path_to_seis_data not specified in config for SDS access")
            self._load_from_sds()

        elif self.config['data_source'].lower() == "fdsn":
            if not self.config['fdsn_server']:
                self.config['fdsn_server'] = "IRIS"  # default server
            self._load_from_fdsn()

        elif self.config['data_source'].lower() == "file":
            if not self.config['path_to_baro_data'] or not self.config['path_to_seis_data']:
                raise ValueError("path_to_baro_data or path_to_seis_data not specified in config for SDS access")
            self._load_from_file()

        else:
            raise ValueError(f"Unknown data source: {self.config['data_source']}")

        # Check if enough data is available
        if len(self.st_baro) == 0:
            print("WARNING: no barometer data available! Aborting...")
            return
        if len(self.st_seis) == 0:
            print("WARNING: no seismometer/rotation data available! Aborting...")
            return

        if self.config['metadata_correction']:
            self._correct_metadata()
        
        # check if data has only nan
        if np.isnan(self.st_baro[0].data).all():
            print("WARNING: barometer data has only nan values! Aborting...")
            return
        
        # check if seismometer/rotation data has only nan
        for tr in self.st_seis:
            if np.isnan(tr.data).all():
                print(f"WARNING: seismometer/rotation data {tr.id} has only nan values! Replacing with zeros.")
                tr.data = np.zeros_like(tr.data)

        # Merge data to avoid nan values
        self.st_baro.merge(method=1)
        self.st_seis.merge(method=1)

        # join to one stream
        self.st = self.st_baro + self.st_seis

        # add hilbert transform
        self.add_hilbert_transform()

        # Handle masked arrays
        self._handle_masked_arrays(self.st)

        # First apply decimation where possible
        self.st = self._decimate_stream(self.st, self.sampling_rate)

         # Check for data problems
        self._check_data_quality()

       # set raw data
        self.st0 = self.st.copy()

        # Stop timer
        stop_timer = timeit.default_timer()
        if self.config['verbose']:
            print(f"\n>Runtime: {round((stop_timer - start_timer)/60, 2)} minutes\n")

    def _correct_metadata(self) -> None:
        """Correct metadata for barometer and seismometer/rotation data."""
        from obspy.clients.fdsn import Client

        # get inventory
        if self.config['baro_inventory']:
            self.baro_inv = read_inventory(self.config['baro_inventory'])
        elif self.config['fdsn_server']:
            self.baro_inv = Client(self.config['fdsn_server']).get_stations(
                network=self.config['network_code'],
                station=self.config['station_code'],
                location=self.config['location_code'],
                channel=self.config['channel_code'],
                level="response",
            )
        else:
            raise ValueError("baro_inventory not specified in config and fdsn_server not specified")
        
        if self.config['seis_inventory']:
            self.seis_inv = read_inventory(self.config['seis_inventory'])
        elif self.config['fdsn_server']:
            self.seis_inv = Client(self.config['fdsn_server']).get_stations(
                network=self.config['network_code'],
                station=self.config['station_code'],
                location=self.config['location_code'],
                channel=self.config['channel_code'],
                level="response",
            )
        else:
            raise ValueError("seis_inventory not specified in config and fdsn_server not specified")

        # remove response
        for tr in self.st_seis + self.st_baro:
            try:
                # for FFBI pressure data
                if "D" == str(tr.stats.channel[1]) and self.config.get('remove_baro_response', False):
                    if self.config.get('pre_filter', False):
                        tr.remove_response(inventory=self.baro_inv, output="DEF", water_level=60, pre_filt=self.config['pre_filter'], plot=False)
                    else:
                        tr.remove_response(inventory=self.baro_inv, output="DEF", water_level=60)
                    # special scaling for FFBI.BDO
                    if "FFBI" in self.config['baro_seed'] and "BDO" in self.config['baro_seed']:
                        tr.data = tr.data * 1e2
                        print(f" >scaling FFBI.BDO barometer data by 1e2 (hPa -> Pa)")
                elif "D" == str(tr.stats.channel[1]) and self.config.get('remove_baro_sensitivity', False):
                    tr.remove_sensitivity(inventory=self.baro_inv)

                # for FFBI pressure data (using manual info)
                elif "D" == str(tr.stats.channel[1]) and self.config.get('remove_baro_gain', False):
                    tr.data = tr.data *1.589e-6 *1e5   # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity = 1 mV/hPa
                elif "F" == str(tr.stats.channel[1]) and self.config.get('remove_baro_gain', False):
                    tr.data = tr.data *1.589e-6 /0.02  # gain=1 sensitivity_reftek=6.28099e5count/V; sensitivity_mb2005=0.02 VPa

                # for ROMY rotations
                elif "J" == str(tr.stats.channel[1]) and self.config.get('remove_seis_sensitivity', False):
                        tr.remove_sensitivity(inventory=self.seis_inv)
                elif "H" == str(tr.stats.channel[1]) and self.config.get('remove_seis_response', False):
                    tr.remove_response(inventory=self.seis_inv, output="ACC", water_level=60)
            except Exception as e:
                print(f">Failed to remove some response: {str(e)}")
                continue

        # make sure its in ZNE
        if "H" == str(self.st_seis[0].stats.channel[1]):
            self.st_seis.rotate("->ZNE", inventory=self.seis_inv)


    def _load_from_sds(self) -> None:
        """
        Load data from local SDS archive.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if self.config['verbose']:
            print("Loading data from SDS archive...")
        
        # Initialize empty streams
        self.st_baro = Stream()
        self.st_seis = Stream()
        
        def _load_seed(seed_info):
            path, net, sta, loc, cha = seed_info
            try:
                return self._read_sds(path, net, sta, loc, cha, self.config['t1'], self.config['t2'])
            except Exception as e:
                print(f">Failed to load {net}.{sta}.{loc}.{cha}: {str(e)}")
                return Stream()
        
        # Prepare all seeds to load
        seeds_to_load = []
        # Add barometer
        net, sta, loc, cha = self.config['baro_seed'].split(".")
        seeds_to_load.append((self.config['path_to_baro_data'], net, sta, loc, cha))
        # Add seismometers
        for seed in self.config['seis_seeds']:
            net, sta, loc, cha = seed.split(".")
            seeds_to_load.append((self.config['path_to_seis_data'], net, sta, loc, cha))
        
        # Load all data in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_load_seed, seed_info) for seed_info in seeds_to_load]
            for i, future in enumerate(as_completed(futures)):
                if i == 0:  # First is barometer
                    self.st_baro += future.result()
                else:  # Rest are seismometers
                    self.st_seis += future.result()

    def _read_sds(self, path_to_archive: str, network: str, station: str, 
                  location: str, channel: str, tbeg: UTCDateTime, 
                  tend: UTCDateTime) -> Stream:
        """
        Read data from SDS archive.
        
        Args:
            path_to_archive: Path to SDS archive
            network: Network code
            station: Station code  
            location: Location code
            channel: Channel code
            tbeg: Start time
            tend: End time
            
        Returns:
            Stream containing requested data
        """
        from obspy.clients.filesystem.sds import Client
        
        if not os.path.exists(path_to_archive):
            raise FileNotFoundError(f">SDS archive path does not exist: {path_to_archive}")
        
        # Initialize SDS client
        client = Client(path_to_archive)
        
        # Get waveforms
        try:
            st = client.get_waveforms(
                network=network,
                station=station, 
                location=location,
                channel=channel,
                starttime=tbeg,
                endtime=tend,
                merge=-1
            )
        except Exception as e:
            print(f">Failed to request {network}.{station}.{location}.{channel}via SDS: {str(e)}")
            st = Stream()
        
        return st

    def _load_from_fdsn(self) -> None:
        """
        Load data from FDSN web service.
        """
        from obspy.clients.fdsn import Client
        
        client = Client(self.config['fdsn_server'])
        
        # Parse network, station, location, channel from seeds
        def parse_seed(seed: str) -> Tuple[str, str, str, str]:
            return seed.split('.')
        
        # Load barometer data
        net, sta, loc, cha = parse_seed(self.config['baro_seed'])
        self.st_baro = client.get_waveforms(
            network=net, station=sta, location=loc, channel=cha,
            starttime=self.config['t1'], endtime=self.config['t2']
        )
        
        # Load seismometer/rotation data
        for seed in self.config['seis_seeds']:
            net, sta, loc, cha = parse_seed(seed)
            self.st_seis += client.get_waveforms(
                network=net, station=sta, location=loc, channel=cha,
                starttime=self.config['t1'], endtime=self.config['t2']
            )

        if self.config['verbose']:
            print(f">Loaded streams from {self.config['fdsn_server']}:")
            print(self.st_baro)
            print(self.st_seis)

    def _load_from_file(self) -> None:
        """Load data from file."""
        from obspy import read
        try:
            _st = read(self.config['path_to_baro_data']).select(channel="*D*")
            net, sta, loc, cha = self.config['baro_seed'].split(".")
            self.st_baro += _st.select(network=net, station=sta, location=loc, channel=cha)
            del _st
        except Exception as e:
            print(f">Failed to load barometric data: {str(e)}")
        try:
            _st = read(self.config['path_to_seis_data'])
            for station in self.config['seis_seeds']:
                net, sta, loc, cha = station.split(".")
                self.st_seis += _st.select(network=net, station=sta, location=loc, channel=cha)
            del _st
        except Exception as e:
            print(f">Failed to load seismic data: {str(e)}")

    def _validate_data_loaded(self) -> None:
        """
        Validate that data is loaded and accessible.
        
        Raises:
            ValueError: If no data is loaded or required streams are missing
        """
        if not hasattr(self, 'st'):
            raise ValueError("No data loaded. Run load_data() first")
        if not hasattr(self, 'st_baro') or len(self.st_baro) == 0:
            raise ValueError("No barometer data loaded")
        if not hasattr(self, 'st_seis') or len(self.st_seis) == 0:
            raise ValueError("No seismometer/rotation data loaded")

    def _validate_time_range(self, tbeg: Optional[UTCDateTime] = None, tend: Optional[UTCDateTime] = None) -> None:
        """
        Validate time range parameters.
        
        Args:
            tbeg: Start time to validate
            tend: End time to validate
            
        Raises:
            ValueError: If time range is invalid
        """
        if tbeg is not None and tend is not None:
            if tbeg >= tend:
                raise ValueError("Start time must be before end time")
        elif self.config['tbeg'] is not None and self.config['tend'] is not None:
            if self.config['tbeg'] >= self.config['tend']:
                raise ValueError("Start time must be before end time")

    def _trim_common_channels(self) -> None:
        """Trim all traces to the shortest trace."""
        npts_min = min([tr.stats.npts for tr in self.st])
        for tr in self.st:
            tr.data = tr.data[:npts_min]
        print(self.st)

    def _check_data_quality(self) -> Dict[str, List[str]]:
        """
        Check data quality and return list of issues.
        """
        # Get seismometer/rotation data
        seis_stream = Stream()
        for seed in self.config['seis_seeds']:
            net, sta, loc, cha = seed.split(".")
            seis_stream += self.st.select(network=net, station=sta, location=loc, channel=cha)

        # Get barometer data
        baro_stream = self.st.select(channel="*D*")

        # Check for gaps
        if len(baro_stream.get_gaps()) > 0:
            print(" >Warning: Gaps found in barometer data")
            baro_stream = baro_stream.merge(method=1)

        # Check for gaps
        if len(seis_stream.get_gaps()) > 0:
            print(" >Warning: Gaps found in seismometer data")
            seis_stream = seis_stream.merge(method=1)

        # Check sampling rates
        baro_sampling = baro_stream[0].stats.sampling_rate
        _channels_smpl = [f"{baro_stream[0].id}={baro_sampling} Hz"]
        for tr in seis_stream:
            if tr.stats.sampling_rate != baro_sampling:
                _channels_smpl.append(f"{tr.id}={tr.stats.sampling_rate} Hz")
        if len(_channels_smpl) > 1:
            print(f" >Warning: Sampling rate mismatch in channels: {_channels_smpl}")

        # Check number of samples
        baro_npts = baro_stream[0].stats.npts
        _channels_npts = [f"{baro_stream[0].id}={baro_npts} samples"]
        for tr in seis_stream:
            if tr.stats.npts != baro_npts:
                _channels_npts.append(f"{tr.id}={tr.stats.npts} samples")
        if len(_channels_npts) > 1:
            print(f" >Warning: Number of samples mismatch in channels: {_channels_npts}")

            # Trim common channels
            self._trim_common_channels()

    def _decimate_stream(self, stream: Stream, target_rate: float) -> Stream:
        """
        Decimate a stream to a lower sampling rate using ObsPy's Decimator.
        
        This method applies a cascade of decimation steps to achieve the target
        sampling rate while maintaining signal quality. It uses ObsPy's Decimator
        which applies an appropriate anti-aliasing filter before decimation.
        
        Args:
            stream: Stream to decimate
            target_rate: Target sampling rate in Hz
            
        Returns:
            Decimated stream
            
        Raises:
            ValueError: If target_rate is higher than current rate or invalid
        """
        # Validate target rate
        if target_rate <= 0:
            raise ValueError("Target sampling rate must be positive")
        
        # Initialize decimator
        decimator = Decimator(
            target_freq=target_rate,
            detrend=True,
            taper=True,
            taper_fraction=0.05,
            filter_before_decim=False,
            filter_after_decim=False,
            filter_freq=(0.0001, 0.3)
        )

        # Create a copy to avoid modifying the original
        st = stream.copy()
        
        # Apply decimation to the stream
        st = decimator.apply_decimation_stream(st)

        return st

    def _handle_masked_arrays(self, stream: Stream, fill_value: float = 0.0) -> None:
        """
        Handle masked arrays in a stream by filling masked values.
        
        Args:
            stream: Stream to process
            fill_value: Value to use for filling masked values
        """
        for tr in stream:
            if isinstance(tr.data, np.ma.MaskedArray):
                tr.data = np.ma.filled(tr.data, fill_value)

    def _preprocess_stream(self, stream: Stream, remove_trend: bool = True, 
                            taper: bool = True, taper_fraction: float = 0.05) -> Stream:
        """
        Apply common preprocessing steps to a stream.
        
        Args:
            stream: Stream to preprocess
            remove_trend: If True, remove mean and linear trend
            taper: If True, apply cosine taper
            taper_fraction: Fraction of data to taper on each end
            
        Returns:
            Preprocessed stream
        """
        if remove_trend:
            stream = stream.detrend('linear')
            stream = stream.detrend('demean')
        
        if taper:
            stream = stream.taper(taper_fraction)
            
        return stream

    @staticmethod
    def store_as_yaml(config: Dict[str, Any], filepath: str) -> None:
        """
        Store a configuration dictionary to a YAML file.
        
        This static method saves a configuration dictionary to a YAML file,
        handling special data types like UTCDateTime appropriately.
        
        Args:
            config: Configuration dictionary to save
            filepath: Path where to save the YAML file
            
        Raises:
            ValueError: If filepath is invalid or directory doesn't exist
            IOError: If file cannot be written
            
        Example:
            >>> bs = baroseis(config)
            >>> baroseis.store_as_yaml(bs.config, 'my_config.yaml')
            # Or directly with a config dictionary:
            >>> config = {'tbeg': UTCDateTime(), 'tend': UTCDateTime()}
            >>> baroseis.store_as_yaml(config, 'my_config.yaml')
        """
        # Convert filepath to Path object
        path = Path(filepath)
        
        # Ensure parent directory exists
        if not path.parent.exists():
            raise ValueError(f"Directory {path.parent} does not exist")
            
        # Create a copy of the config to modify
        config_to_save = dict(config)
        
        # Convert UTCDateTime objects to strings
        for key in ['tbeg', 'tend']:
            if key in config_to_save and config_to_save[key] is not None:
                if isinstance(config_to_save[key], UTCDateTime):
                    config_to_save[key] = str(config_to_save[key])
                
        # Save to YAML
        try:
            with open(path, 'w') as f:
                yaml.dump(config_to_save, f, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Failed to write YAML file: {str(e)}")
            
    @staticmethod
    def load_from_yaml(filepath: str) -> Dict[str, Any]:
        """
        Load a configuration dictionary from a YAML file.
        
        This static method loads a configuration from a YAML file,
        handling special data types like UTCDateTime appropriately.
        
        Args:
            filepath: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary with converted data types
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            ValueError: If the YAML file is invalid or contains invalid time formats
            
        Example:
            >>> # Load config and create new instance
            >>> config = baroseis.load_from_yaml('my_config.yaml')
            >>> bs = baroseis(config)
            >>> 
            >>> # Or update existing instance
            >>> bs.config.update(baroseis.load_from_yaml('my_config.yaml'))
        """
        # Convert filepath to Path object
        path = Path(filepath)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Configuration file {filepath} not found")
            
        # Load YAML
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load YAML file: {str(e)}")
            
        # Convert time strings to UTCDateTime objects
        for key in ['tbeg', 'tend']:
            if key in config and config[key] is not None:
                try:
                    config[key] = UTCDateTime(config[key])
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid time format for {key}: {str(e)}")
        
        # Convert pre_filter list to tuple if it exists
        if 'pre_filter' in config and isinstance(config['pre_filter'], list):
            config['pre_filter'] = tuple(config['pre_filter'])
                    
        return config

    def filter_data(self, fmin: Optional[float] = None, fmax: Optional[float] = None) -> None:
        """
        Apply preprocessing and filtering to the data streams.
        
        This method performs several preprocessing steps on the data:
        1. Removes mean and linear trend from all traces
        2. Applies a cosine taper (5% on each end)
        3. Applies bandpass, lowpass, or highpass filtering based on provided frequency limits
        4. Checks data quality after filtering
        
        The filtering is done using a zero-phase Butterworth filter with 4 corners.
        The filter type is automatically selected based on the provided frequency limits:
        - If both fmin and fmax: bandpass filter
        - If only fmin: lowpass filter
        - If only fmax: highpass filter
        
        Args:
            fmin (Optional[float]): Lower frequency corner in Hz. If provided, overrides
                                   config['fmin']. Use None to skip lowpass filtering.
            fmax (Optional[float]): Upper frequency corner in Hz. If provided, overrides
                                   config['fmax']. Use None to skip highpass filtering.
        
        Raises:
            ValueError: If no data is loaded (self.st is not initialized)
            ValueError: If frequency limits are invalid (e.g., fmin > fmax)
            Exception: If filtering fails for any trace
        
        Notes:
            - The method operates on self.st (combined stream)
            - Original data can be accessed in self.st0 (if saved during load_data)
            - Filtering is done in-place, modifying the stream data
            - Data quality checks include:
                - Sampling rate consistency
                - Gap detection
                - Number of samples consistency
        
        Example:
            >>> bs = baroseis(config)
            >>> bs.load_data()
            >>> # Apply bandpass filter between 0.1 and 10 mHz
            >>> bs.filter_data(fmin=0.0001, fmax=0.01)
            >>> # Apply only lowpass filter at 10 mHz
            >>> bs.filter_data(fmax=0.01)
            >>> # Apply only highpass filter at 0.1 mHz
            >>> bs.filter_data(fmin=0.0001)
        """
        # Validate data is loaded
        self._validate_data_loaded()
        
        # Validate frequency parameters
        if fmin is not None and fmax is not None and fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if fmin is not None and fmin < 0:
            raise ValueError(f"fmin ({fmin}) must be non-negative")
        if fmax is not None and fmax < 0:
            raise ValueError(f"fmax ({fmax}) must be non-negative")
            
        # Check if any traces have sampling rate too low for fmax
        if fmax is not None:
            nyquist_freqs = [tr.stats.sampling_rate/2 for tr in self.st]
            min_nyquist = min(nyquist_freqs)
            if fmax >= min_nyquist:
                raise ValueError(f"fmax ({fmax}) must be less than Nyquist frequency ({min_nyquist})")
        
        # Update fmin and fmax if provided
        if fmin is not None:
            self.config['fmin'] = fmin
        if fmax is not None:
            self.config['fmax'] = fmax

        # Remove mean and trend
        if self.config['verbose']:
            print("Removing mean and trend...")
        self.st = self.st.detrend('linear')
        self.st = self.st.detrend('demean')
        self.st = self.st.taper(0.05)

        # Apply bandpass filter if configured
        filter_type = "bandpass" if fmin is not None and fmax is not None else \
                     "lowpass" if fmax is not None else \
                     "highpass" if fmin is not None else None
        if filter_type:
            if self.config['verbose']:
                print(f"Applying {filter_type} filter: {self.config['fmin']}-{self.config['fmax']} Hz")
        if self.config['fmin'] is not None and self.config['fmax'] is not None:
            self.st = self.st.filter('bandpass',
                        freqmin=self.config['fmin'],
                        freqmax=self.config['fmax'],
                        corners=4,
                        zerophase=True)
        elif self.config['fmin'] is not None:
            self.st = self.st.filter('lowpass',
                        freq=self.config['fmin'],
                        corners=4,
                        zerophase=True)
        elif self.config['fmax'] is not None:
            self.st = self.st.filter('highpass',
                        freq=self.config['fmax'],
                        corners=4,
                        zerophase=True)

        # Check for data problems
        self._check_data_quality()

    def write_to_file(self, filename: str, datapath: Optional[str] = None, format: str = "MSEED") -> None:
        """
        Write stream data to a file.
        
        Args:
            filename: Name of the output file
            datapath: Optional path where to save the file. If None, uses current directory
            format: File format (default: "MSEED")
            
        Raises:
            ValueError: If no data is loaded or path is invalid
            IOError: If file cannot be written
        """
        # Validate data is loaded
        self._validate_data_loaded()
        
        # Handle path
        if datapath is not None:
            # Create directory if it doesn't exist
            path = Path(datapath)
            path.mkdir(parents=True, exist_ok=True)
            filepath = path / filename
        else:
            filepath = Path(filename)
            
        try:
            # Write data
            self.st.write(str(filepath), format=format)
        except Exception as e:
            raise IOError(f"Failed to write file {filepath}: {str(e)}")
            
    def write_to_sds(self, stream: Optional[Stream] = None, sds_path: Optional[str] = None) -> None:
        """
        Write stream to SDS file structure.
        
        Args:
            stream: Stream to write (defaults to self.st if None)
            sds_path: Root path of SDS archive (defaults to config sds_path if None)
        """
        if stream is None:
            stream = self.st
        
        if sds_path is None:
            sds_path = self.config['sds_path']
        
        if not os.path.exists(sds_path):
            raise ValueError(f"SDS path does not exist: {sds_path}")
        
        try:
            # Initialize SDS client
            client = SDSClient(sds_path)
            
            # Write each trace to SDS structure
            for tr in stream:
                # Get time info
                year = str(tr.stats.starttime.year)
                julday = "%03d" % tr.stats.starttime.julday
                
                # Create directory structure
                net_dir = os.path.join(sds_path, year, tr.stats.network)
                sta_dir = os.path.join(net_dir, tr.stats.station)
                cha_dir = os.path.join(sta_dir, f"{tr.stats.channel}.D")
                
                # Create directories if they don't exist
                for directory in [net_dir, sta_dir, cha_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                
                # Write trace
                stream.write(tr, format="MSEED")
                
                if self.config['verbose']:
                    print(f"Written: {tr.id} for {year}-{julday}")
                
        except Exception as e:
            print(f"Error writing to SDS: {str(e)}")

    def integrate_data(self, method: str = "spline", channels=['JZ','JN','JE']) -> None:
        """
        Integrate the data streams.
        
        Args:
            method: Integration method ('spline' or 'cumtrapz')
        """
        if self.config['verbose']:
            print(f" >Integrating {channels}")
        
        for ch in channels:
            try:
                tr = self.st.select(channel=f'*{ch}')[0].copy()
            except Exception as e:
                print(f" >No data found for channel {ch}: {str(e)}")
                continue

            # remove mean and trend
            tr = tr.detrend('linear').detrend('demean')
            # integrate
            tr.data = tr.integrate(method=method).data
            # add standard
            tr.stats.standard = 'Integral'
            # remove mean and trend
            # tr = tr.detrend('linear').detrend('demean')
            # adopt component names north to east and east to north
            if tr.stats.channel[2] == "N":
                tr.stats.channel = tr.stats.channel[0] + "E" + tr.stats.channel[2:]
            elif tr.stats.channel[2] == "E":
                tr.stats.channel = tr.stats.channel[0] + "N" + tr.stats.channel[2:]
                # compensate for rotation definition
                tr.data *= -1
            # change channel name by replacing central character with "A"
            tr.stats.channel = tr.stats.channel[0] + "A" + tr.stats.channel[2:]
            # add to stream
            self.st += tr

    def add_pressure_gradient(self, brmy_seed: str = "BW.BRMY..BDO") -> None:
        """
        Add pressure gradient data using BRMY barometer data.
        
        This function:
        1. Loads BRMY pressure data
        2. Calculates pressure gradient using FFBI and BRMY data
        3. Adds gradient traces to the stream
        
        Args:
            brmy_seed: SEED ID for BRMY barometer data
        """
        if not hasattr(self, 'st_baro') or len(self.st_baro) == 0:
            raise ValueError("No barometer data loaded. Run load_data() first")
        
        # Load BRMY data
        try:
            st_brmy = Stream()
            net, sta, loc, cha = brmy_seed.split('.')
            
            if self.config['data_source'].lower() == "sds":
                if not self.config['path_to_baro_data']:
                    raise ValueError("path_to_baro_data not specified in config for SDS access")
                
                st_brmy = self._read_sds(
                    self.config['path_to_baro_data'],
                    net, sta, loc, cha,
                    self.config['t1'],
                    self.config['t2']
                )
            else:
                raise ValueError("Only SDS data source is currently supported for BRMY data")
            
            if len(st_brmy) == 0:
                raise ValueError(f"No BRMY data found for {brmy_seed}")
            
        except Exception as e:
            raise ValueError(f"Could not load BRMY data: {str(e)}")
        
        # Add traces to stream
        self.st += st_brmy
        
        if self.config['verbose']:
            print(f"Added pressure gradient data:")

    def add_hilbert_transform(self, overwrite: bool = True) -> None:
        """
        Add Hilbert transform to the data streams.

        Args:
            overwrite: If True, overwrite existing Hilbert transform
        """

        def _compute_hilbert_transform(self) -> Stream:
            """
            Compute Hilbert transform for the barometer data stream.
            
            This function computes the Hilbert transform of the barometer data,
            which provides the quadrature component of the signal. This is useful
            for analyzing phase relationships and envelope detection.
            
            Returns:
                Stream containing the Hilbert transformed barometer data with
                channel code modified to end in 'H' instead of 'O'
                
            Raises:
                ValueError: If no barometer data is found or if multiple traces are found
            """
            # from scipy.signal import hilbert
            from obspy.signal.filter import hilbert

            bdh = self.st.select(channel="*D*").copy()
            if len(bdh) == 0:
                raise ValueError("No data found for channel *DO in stream")
            if len(bdh) == 1:
                for tr in bdh:
                    tr.data = hilbert(tr.data)
                    tr.stats.standard = 'Hilbert'
                    tr.stats.channel = tr.stats.channel[:-1] + "H"
                    return bdh
            else:
                raise ValueError("Multiple data found for channel *DO in stream. Aborting...")

        # check if Hilbert transform already exists
        if not hasattr(self, 'st_baro_hilbert'):
            self.st_baro_hilbert = None

        if overwrite and self.st_baro_hilbert is None:

            if self.config['verbose']:
                print("Creating new Hilbert transform data...")

            # create attribute
            self.st_baro_hilbert = _compute_hilbert_transform(self)

            self.st += self.st_baro_hilbert

        elif overwrite and self.st_baro_hilbert is not None:

            if self.config['verbose']:
                print("Overwriting existing Hilbert transform data...")

            # remove existing Hilbert transform data from stream
            for _tr in self.st.select(component="H"):
                self.st.remove(_tr)

            # compute Hilbert transform
            self.st_baro_hilbert = _compute_hilbert_transform(self)

            # join to one stream
            self.st += self.st_baro_hilbert
        else:
            if self.config['verbose']:
                print("Hilbert transform already exists. To overwrite set: overwrite=True.")

    def variance_reduction(self, arr1: ndarray, arr2: ndarray) -> float:
        """
        Compute variance reduction between two arrays.
        
        Args:
            arr1: Reference array (original data)
            arr2: Array to compare against (usually residual)
            
        Returns:
            Percentage of variance reduction rounded to 2 decimal places
            
        Notes:
            Variance reduction is calculated as: ((var(arr1) - var(arr2)) / var(arr1)) * 100
            A positive value indicates a reduction in variance
        """
        from numpy import var
        return round((var(arr1) - var(arr2)) / var(arr1) * 100, 2)

    # =========================================================================
    # Analysis and Prediction Methods
    # =========================================================================

    def predict_tilt_from_pressure(self, method: str = "least_squares", 
                                 zero_intercept: bool = False, verbose: bool = True,
                                 channel_type: str = "J") -> None:
        """
        Predict tilt, rotation rate, or acceleration from pressure data using regression analysis.
        
        This method performs advanced regression analysis to predict and remove pressure-induced
        signals from rotation, tilt, or acceleration measurements. It uses both the raw pressure
        signal and its Hilbert transform as predictors to account for phase relationships.

        The method supports multiple regression techniques, from simple least squares to robust
        methods like TheilSen and RANSAC, making it suitable for data with outliers or
        non-linear relationships.

        Args:
            method: Regression method to use:
                   - 'least_squares': Standard least squares regression (default)
                   - 'theilsen': Theil-Sen regression (robust to outliers)
                   - 'ransac': RANSAC regression (robust to outliers)
                   - 'odr': Orthogonal Distance Regression
                   - 'ols': Standard least squares regression (alternative)
            zero_intercept: If True, force regression through origin. Default: False
            verbose: If True, print detailed progress information. Default: True
            channel_type: Type of channel to analyze:
                         - 'J': Rotation rate (nrad/s)
                         - 'A': Tilt (nrad)
                         - 'H': Acceleration (nm/s²)
        
        Raises:
            ValueError: If no data is loaded or Hilbert transform is not computed
            ValueError: If method is not one of the supported regression methods
            ValueError: If channel_type is not 'J', 'A', or 'H'
            TypeError: If input parameters have incorrect types
            RuntimeError: If processing fails for any component
        
        
        Notes:
            - The method requires running add_hilbert_transform() first
            - Results are stored in class attributes and added to the stream:
                * p_coefficient: Regression coefficients for pressure
                * h_coefficient: Regression coefficients for Hilbert transform
                * New traces with location code 'PP' in self.st
            - Coefficients are stored in natural units (e.g., nrad/s/hPa)
            - Variance reduction is calculated and printed if verbose=True
            - Each component (N,E,Z) is processed independently
            - The method automatically handles data scaling and units
        """
        # Validate input parameters
        if method.lower() not in ['least_squares', 'theilsen', 'ransac', 'odr', 'ols']:
            raise ValueError("Method must be 'least_squares' or one of: 'theilsen', 'ransac', 'odr', 'ols'")
            
        if channel_type.upper() not in ['J', 'A', 'H']:
            raise ValueError("channel_type must be 'J' (rotation rate), 'A' (tilt), or 'H' (acceleration)")
            
        # Validate data is loaded and has required attributes
        self._validate_data_loaded()
        
        if not hasattr(self, 'st_baro_hilbert'):
            raise ValueError("Hilbert transform not computed. Run add_hilbert_transform() first")

        # Remove PP traces from stream
        for tr in self.st:
            if tr.stats.location == 'PP':
                self.st.remove(tr)

        # Get required traces
        try:
            tr_p = self.st.select(channel="*DO").copy()[0]
            tr_h = self.st.select(channel="*DH").copy()[0]
        except IndexError:
            raise ValueError("Could not find required pressure or Hilbert transform traces")
            
        # Initialize dictionaries to store coefficients
        self.p_coefficient = {}
        self.h_coefficient = {}
        
        # Get pressure and Hilbert transform
        tr_p = self.st.select(channel="*DO").copy()[0]
        tr_h = self.st.select(channel="*DH").copy()[0]
        
        # Process each component
        for comp in ['N', 'E', 'Z']:
            if verbose:
                print(f"\nComponent {comp}:")
            
            try:
                tr_rot = self.st.select(channel=f"*{channel_type}{comp}").copy()[0]
                
                if method.lower() == "least_squares":
                    # Original least squares method
                    A = np.vstack([tr_p.data, tr_h.data]).T
                    ratio = np.linalg.lstsq(A, tr_rot.data, rcond=None)[0]
                    pred_data = ratio[0] * tr_p.data + ratio[1] * tr_h.data
                    
                    # Store ratios
                    self.p_coefficient[comp] = ratio[0]
                    self.h_coefficient[comp] = ratio[1]
                else:
                    # Create dataframe for regression
                    df = pd.DataFrame({
                        f'{channel_type}{comp}': tr_rot.data,
                        'P': tr_p.data,
                        'H': tr_h.data,
                        'time': tr_p.times()
                    })

                    # Use regression method
                    reg_result = self.regression(
                        data=df,
                        target=f'{channel_type}{comp}',
                        features=['P', 'H'],
                        reg_type=method,
                        zero_intercept=zero_intercept,
                        verbose=verbose
                    )
                    
                    # Get prediction and ratios
                    pred_data = reg_result['predicted']
                    self.p_coefficient[comp] = reg_result.get('coef', [0, 0])[0]
                    self.h_coefficient[comp] = reg_result.get('coef', [0, 0])[1]


                # Create predicted trace
                tr_pred = tr_rot.copy()
                tr_pred.stats.location = "PP"  # Set location to PP for predicted data
                tr_pred.data = pred_data

                # Add to stream
                self.st += tr_pred

                # Set factor for printing coefficients
                factor = 1e11

                if verbose:
                    var_red = self.variance_reduction(tr_rot.data, tr_rot.data - pred_data)
                    if channel_type == "A":
                        print(f"P coefficient: {self.p_coefficient[comp]*factor:.3f} nrad/hPa")
                        print(f"H coefficient: {self.h_coefficient[comp]*factor:.3f} nrad/hPa")
                    elif channel_type == "H":
                        print(f"P coefficient: {self.p_coefficient[comp]*factor:.3f} nm/s²/hPa")
                        print(f"H coefficient: {self.h_coefficient[comp]*factor:.3f} nm/s²/hPa")
                    elif channel_type == "J":
                        print(f"P coefficient: {self.p_coefficient[comp]*factor:.3f} nrad/s/hPa")
                        print(f"H coefficient: {self.h_coefficient[comp]*factor:.3f} nrad/s/hPa")
                    else:
                        print(f" -> Warning: Unknown channel type: {channel_type}")
                    print(f"Variance reduction: {var_red:.1f}%")
    
            except Exception as e:
                print(f"Could not process component {comp}: {str(e)}")
                continue

    def model_tilt_from_pressure(self, method: str = "least_squares", 
                                zero_intercept: bool = False, verbose: bool = True,
                                channel_type: str = "J", out: bool = False) -> None:
        """
        Predict tilt, rotation rate, or acceleration from pressure data using regression analysis
        with derivatives of both pressure and Hilbert transform.
        
        This method performs advanced regression analysis to predict and remove pressure-induced
        signals from rotation, tilt, or acceleration measurements. It uses four predictors:
        1. Raw pressure signal
        2. Hilbert transform of pressure
        3. Derivative of pressure signal
        4. Derivative of Hilbert transform

        The method supports multiple regression techniques, from simple least squares to robust
        methods like TheilSen and RANSAC, making it suitable for data with outliers or
        non-linear relationships.

        Args:
            method: Regression method to use:
                   - 'least_squares': Standard least squares regression (default)
                   - 'theilsen': Theil-Sen regression (robust to outliers)
                   - 'ransac': RANSAC regression (robust to outliers)
                   - 'odr': Orthogonal Distance Regression
                   - 'ols': Standard least squares regression (alternative)
            zero_intercept: If True, force regression through origin. Default: False
            verbose: If True, print detailed progress information. Default: True
            channel_type: Type of channel to analyze:
                         - 'J': Rotation rate (nrad/s)
                         - 'A': Tilt (nrad)
                         - 'H': Acceleration (nm/s²)
            out: If True, return dictionary containing model data
        Raises:
            ValueError: If no data is loaded or Hilbert transform is not computed
            ValueError: If method is not one of the supported regression methods
            ValueError: If channel_type is not 'J', 'A', or 'H'
            TypeError: If input parameters have incorrect types
            RuntimeError: If processing fails for any component
        
        
        Notes:
            - The method requires running add_hilbert_transform() first
            - Results are stored in class attributes and added to the stream:
                * p_coefficient: Regression coefficients for pressure
                * h_coefficient: Regression coefficients for Hilbert transform
                * dp_coefficient: Regression coefficients for pressure derivative
                * dh_coefficient: Regression coefficients for Hilbert derivative
                * New traces with location code 'PP' in self.st
            - Coefficients are stored in natural units (e.g., nrad/s/hPa)
            - Variance reduction is calculated and printed if verbose=True
            - Each component (N,E,Z) is processed independently
            - The method automatically handles data scaling and units
        """
        # Validate input parameters
        if method.lower() not in ['least_squares', 'theilsen', 'ransac', 'odr', 'ols']:
            raise ValueError("Method must be 'least_squares' or one of: 'theilsen', 'ransac', 'odr', 'ols'")
            
        if channel_type.upper() not in ['J', 'A', 'H']:
            raise ValueError("channel_type must be 'J' (rotation rate), 'A' (tilt), or 'H' (acceleration)")
            
        # Validate data is loaded and has required attributes
        self._validate_data_loaded()
        
        if not hasattr(self, 'st_baro_hilbert'):
            raise ValueError("Hilbert transform not computed. Run add_hilbert_transform() first")
            
        # Get required traces
        try:
            tr_p = self.st.select(channel="*DO*").copy()[0]
            tr_h = self.st.select(channel="*DH").copy()[0]
        except IndexError:
            raise ValueError("Could not find required pressure or Hilbert transform traces")
            
        # Initialize dictionaries to store coefficients
        self.p_coefficient = {}
        self.h_coefficient = {}
        self.dp_coefficient = {}
        self.dh_coefficient = {}

        # Compute derivatives
        tr_dp = tr_p.copy().differentiate()
        # tr_dp.data = np.gradient(tr_p.data, tr_p.stats.delta)
        tr_dp.stats.location = 'DD'  # Mark as derivative
        
        tr_dh = tr_h.copy().differentiate()
        # tr_dh.data = np.gradient(tr_h.data, tr_h.stats.delta)
        tr_dh.stats.location = 'DD'  # Mark as derivative
        
        # Remove PP traces from stream
        for tr in self.st:
            if tr.stats.location == 'PP':
                self.st.remove(tr)
        
        # Initialize dictionary to store model data
        model_data = {}

        # Process each component
        for comp in ['N', 'E', 'Z']:
            if verbose:
                print(f"\nComponent {comp} (with derivatives):")
            
            try:
                tr_rot = self.st.select(channel=f"*{channel_type}{comp}").copy()[0]
                
                if method.lower() == "least_squares":
                    # Original least squares method with 4 predictors
                    A = np.vstack([tr_p.data, tr_h.data, tr_dp.data, tr_dh.data]).T
                    ratio = np.linalg.lstsq(A, tr_rot.data, rcond=None)[0]
                    pred_data = (ratio[0] * tr_p.data + ratio[1] * tr_h.data + 
                               ratio[2] * tr_dp.data + ratio[3] * tr_dh.data)
                    
                    # Store ratios
                    self.p_coefficient[comp] = ratio[0]
                    self.h_coefficient[comp] = ratio[1]
                    self.dp_coefficient[comp] = ratio[2]
                    self.dh_coefficient[comp] = ratio[3]

                    model_data[comp] = pred_data

                else:
                    # Create dataframe for regression with 4 predictors
                    df = pd.DataFrame({
                        f'{channel_type}{comp}': tr_rot.data,
                        'P': tr_p.data,
                        'H': tr_h.data,
                        'DP': tr_dp.data,
                        'DH': tr_dh.data,
                        'time': tr_p.times()
                    })

                    # Use regression method
                    reg_result = self.regression(
                        data=df,
                        target=f'{channel_type}{comp}',
                        features=['P', 'H', 'DP', 'DH'],
                        reg_type=method,
                        zero_intercept=zero_intercept,
                        verbose=verbose
                    )
                    
                    # Get prediction and ratios
                    pred_data = reg_result['predicted']
                    coef = reg_result.get('coef', [0, 0, 0, 0])
                    self.p_coefficient[comp] = coef[0]
                    self.h_coefficient[comp] = coef[1]
                    self.dp_coefficient[comp] = coef[2]
                    self.dh_coefficient[comp] = coef[3]

                    model_data[comp] = pred_data

                # Create predicted trace
                tr_pred = tr_rot.copy()
                tr_pred.stats.location = "PP"  # Set location to PP for predicted data
                tr_pred.data = pred_data

                # Add to stream
                self.st += tr_pred

                # Set factor for printing coefficients
                factor = 1e11

                if verbose:
                    var_red = self.variance_reduction(tr_rot.data, tr_rot.data - pred_data)
                    if channel_type == "A":
                        print(f"P coefficient: {self.p_coefficient[comp]*factor:.3f} nrad/hPa")
                        print(f"H coefficient: {self.h_coefficient[comp]*factor:.3f} nrad/hPa")
                        print(f"DP coefficient: {self.dp_coefficient[comp]*factor:.3f} nrad/hPa/s")
                        print(f"DH coefficient: {self.dh_coefficient[comp]*factor:.3f} nrad/hPa/s")
                    elif channel_type == "H":
                        print(f"P coefficient: {self.p_coefficient[comp]*factor:.3f} nm/s²/hPa")
                        print(f"H coefficient: {self.h_coefficient[comp]*factor:.3f} nm/s²/hPa")
                        print(f"DP coefficient: {self.dp_coefficient[comp]*factor:.3f} nm/s²/hPa/s")
                        print(f"DH coefficient: {self.dh_coefficient[comp]*factor:.3f} nm/s²/hPa/s")
                    elif channel_type == "J":
                        print(f"P coefficient: {self.p_coefficient[comp]*factor:.3f} nrad/s/hPa")
                        print(f"H coefficient: {self.h_coefficient[comp]*factor:.3f} nrad/s/hPa")
                        print(f"DP coefficient: {self.dp_coefficient[comp]*factor:.3f} nrad/s/hPa/s")
                        print(f"DH coefficient: {self.dh_coefficient[comp]*factor:.3f} nrad/s/hPa/s")
                    else:
                        print(f" -> Warning: Unknown channel type: {channel_type}")
                    print(f"Variance reduction: {var_red:.1f}%")
    
            except Exception as e:
                print(f"Could not process component {comp}: {str(e)}")
                continue
        
        if out:
            return {
                'N': {
                    'p_coefficient': self.p_coefficient['N'],
                    'h_coefficient': self.h_coefficient['N'],
                    'dp_coefficient': self.dp_coefficient['N'],
                    'dh_coefficient': self.dh_coefficient['N'],
                    'time': tr_p.times(),
                    'data': model_data['N']
                },
                'E': {
                    'p_coefficient': self.p_coefficient['E'],
                    'h_coefficient': self.h_coefficient['E'],
                    'dp_coefficient': self.dp_coefficient['E'],
                    'dh_coefficient': self.dh_coefficient['E'],
                    'time': tr_p.times(),
                    'data': model_data['E']
                },
                'Z': {
                    'p_coefficient': self.p_coefficient['Z'],
                    'h_coefficient': self.h_coefficient['Z'],
                    'dp_coefficient': self.dp_coefficient['Z'],
                    'dh_coefficient': self.dh_coefficient['Z'],
                    'time': tr_p.times(),
                    'data': model_data['Z']
                }
            }

    def compute_cross_correlation(self, 
                                win_time_s: Optional[float] = None,
                                overlap: Optional[float] = None,
                                plot: Optional[bool] = None) -> Dict:
        """
        Compute cross-correlation between barometer and seismometer/rotation components.

        Args:
            win_time_s: Window length in seconds (overrides config)
            overlap: Window overlap fraction (overrides config)
            plot: Generate plot if True (overrides config)

        Returns:
            Dictionary containing correlation results
        """
        # Use parameters from config if not provided
        win_time_s = win_time_s or self.config['win_length_sec']
        overlap = overlap or self.config['overlap']
        plot = plot if plot is not None else self.config['plot']
        
        from functions.cross_correlation_function_windows import __cross_correlation_function_windows

        results = {}
        
        for tr in self.st_seis:
            times, ccf, lags, shifts, maxima = __cross_correlation_function_windows(
                self.st_baro[0].data,
                tr.data,
                self.st_baro[0].stats.delta,
                win_time_s,
                overlap=overlap
            )
            
            results[tr.stats.channel] = {
                'times': times,
                'ccf': ccf,
                'lags': lags,
                'shifts': shifts,
                'maxima': maxima
            }

        if plot:
            self._plot_cross_correlations(results)

        return results

    def compute_coherence(self, window_sec: float = 3600.0, overlap: float = 0.5,
                         smooth_points: int = 31, baro_channel: str = "BDX",
                         channels: Optional[List[str]] = None) -> Dict[str, Dict[str, Union[ndarray, float, str]]]:
        """
        Compute coherence between barometer and seismometer/rotation components.
        
        Args:
            window_sec: Window length in seconds for coherence calculation
            overlap: Window overlap fraction (0-1)
            smooth_points: Number of points for coherence smoothing
            baro_channel: Channel code for barometer (e.g. "BDX", "LD*")
            channels: Optional list of channel codes to compute coherence for.
                     If None, computes for all available rotation/seismometer channels.
                     Example: ['BJZ', 'BJN', 'BJE'] for specific rotation channels
            
        Returns:
            Dictionary containing frequencies and coherence for each component.
            Keys are channel IDs, values are dictionaries containing:
            - frequencies: Array of frequencies
            - coherence: Array of coherence values
            - window_sec: Window length used
            - overlap: Overlap fraction used
            - baro_id: ID of barometer channel used
        """
        # Validate input parameters
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0 and 1")
        if smooth_points < 0:
            raise ValueError("smooth_points must be non-negative")
            
        # Validate data is loaded
        self._validate_data_loaded()
        
        # Import required functions
        try:
            from scipy.signal import coherence, savgol_filter
            from scipy.signal.windows import hann
        except ImportError as e:
            raise ImportError("Required scipy functions not available") from e

        # Get barometer trace
        try:
            baro_tr = self.st.select(channel=f"*{baro_channel}*").copy()
            if not baro_tr:
                raise ValueError(f"No barometer channel matching '*{baro_channel}*' found")
            baro_tr = baro_tr[0]
        except Exception as e:
            raise ValueError(f"Error accessing barometer data: {str(e)}")
        
        # Calculate window parameters
        npts = int(window_sec * baro_tr.stats.sampling_rate)
        if npts > len(baro_tr.data):
            raise ValueError(f"Window length ({window_sec}s) longer than data duration")
        noverlap = int(npts * overlap)
        
        # Create output dictionary
        coherence_dict = {}
        # Get traces to process
        if channels is not None:
            # Use provided channel list
            traces = []
            for channel in channels:
                tr = self.st.select(channel=f"*{channel}*").copy()
                if tr:
                    traces.extend(tr)
                else:
                    print(f"Warning: No data found for channel {channel}")
        else:
            # Default to all rotation/seismometer channels
            traces = self.st.select(channel="*J*").copy()

        # Compute coherence for selected components
        for tr in traces:
            # Skip if same as barometer
            if tr.id == baro_tr.id:
                continue
                        
            # Check sampling rates match
            if tr.stats.sampling_rate != baro_tr.stats.sampling_rate:
                print(f"Warning: Sampling rate mismatch for {tr.id}, skipping")
                continue
            
            # Check for NaN values
            if np.isnan(tr.data).any() or np.isnan(baro_tr.data).any():
                print(f"Warning: NaN values found for {tr.id}, skipping")
                continue
            
            try:
                # Compute coherence
                freq, coh = coherence(
                    baro_tr.data, tr.data,
                    fs=tr.stats.sampling_rate,
                    nperseg=npts,
                    noverlap=noverlap,
                    window=hann(npts)
                )
                
                # Smooth coherence if requested
                if smooth_points > 0:
                    if len(coh) > smooth_points:
                        coh = savgol_filter(coh, smooth_points, 3)
                    else:
                        print(f"Warning: Not enough points for smoothing {tr.id}")
                
                coherence_dict[tr.id] = {
                    'frequencies': freq,
                    'coherence': coh,
                    'window_sec': window_sec,
                    'overlap': overlap,
                    'baro_id': baro_tr.id,
                    'seis_id': [tr.id for tr in traces]
                }
                
            except Exception as e:
                print(f"Error computing coherence for {tr.id}: {str(e)}")
                continue
        
        if not coherence_dict:
            raise ValueError("No valid coherence results computed")
        
        return coherence_dict

    def compute_cwt(self, plot: bool = True) -> Dict:
        """
        Compute continuous wavelet transform for barometer and seismometer data.

        Args:
            plot: Generate plot if True

        Returns:
            Dictionary containing CWT results
        """
        from functions.compute_cwt import __compute_cwt
        
        results = {}
        
        # Compute CWT for barometer
        baro_cwt = __compute_cwt(
            self.st_baro[0].times(),
            self.st_baro[0].data,
            self.st_baro[0].stats.delta,
            plot=False
        )
        results['barometer'] = baro_cwt
        
        # Compute CWT for each seismometer component
        for tr in self.st_seis:
            cwt = __compute_cwt(
                tr.times(),
                tr.data,
                tr.stats.delta,
                plot=False
            )
            results[tr.stats.channel] = cwt

        if plot:
            self._plot_cwt(results)

        return results

    def compute_fft(self, signal_in: ndarray, dt: float, window: Optional[str] = None) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Calculate FFT spectrum of a time series.
        
        Args:
            signal_in: Input time series
            dt: Time step (sampling interval)
            window: Window function name (e.g., 'hanning', 'hamming')
        
        Returns:
            Tuple containing:
            - magnitude: FFT magnitude spectrum
            - frequencies: Frequency array
            - phase: Phase spectrum
        """
        from scipy.fft import fft, fftfreq
        from scipy import signal
        
        # Determine length of input time series
        n = len(signal_in)
        
        # Apply window if specified
        if window:
            win = signal.get_window(window, n)
            spectrum = fft(signal_in * win)
        else:
            spectrum = fft(signal_in)
        
        # Calculate frequency array
        frequencies = fftfreq(n, d=dt)
        
        # Calculate magnitude spectrum
        magnitude = abs(spectrum) * 2.0 / n
        
        # Calculate phase spectrum
        phase = np.angle(spectrum, deg=False)
        
        # Return positive frequencies only
        return magnitude[0:n//2], frequencies[0:n//2], phase[0:n//2]

    def regression(self, data: pd.DataFrame, target: str, features: List[str], 
                  reg_type: str = "theilsen", zero_intercept: bool = False, 
                  verbose: bool = False) -> Dict[str, Union[ndarray, float, Any]]:
        """
        Perform regression analysis.
        
        Args:
            data: DataFrame containing the data
            target: Name of target variable column
            features: List of feature column names
            reg_type: Type of regression ('ols', 'ransac', 'theilsen', 'odr')
            zero_intercept: Force regression through zero
            verbose: Print results if True
        
        Returns:
            Dictionary containing regression results
        """
        from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
        from sklearn.metrics import r2_score
        import numpy as np

        # Validate regression method
        valid_methods = ['ols', 'ransac', 'theilsen', 'odr']
        if reg_type.lower() not in valid_methods:
            raise ValueError(f"Invalid regression method. Must be one of {valid_methods}")

        # Prepare X and y data
        X = data[features].values
        y = data[target].values

        # Initialize model based on regression type
        if reg_type.lower() == 'ols':
            model = LinearRegression(
                fit_intercept=not zero_intercept
            )
        elif reg_type.lower() == 'ransac':
            model = RANSACRegressor(
                estimator=LinearRegression(fit_intercept=not zero_intercept),
                random_state=42
            )
        elif reg_type.lower() == 'theilsen':
            model = TheilSenRegressor(
                fit_intercept=not zero_intercept,
                random_state=42
            )
        elif reg_type.lower() == 'odr':
            from scipy import odr
            
            # Define linear function for ODR
            def f(B: ndarray, x: ndarray) -> ndarray:
                return B[0] * x[:, 0] + B[1] * x[:, 1] + (0 if zero_intercept else B[2])
            
            # Create ODR model
            linear = odr.Model(f)
            
            # Create ODR data object
            mydata = odr.RealData(X, y)
            
            # Set initial parameter guess
            if zero_intercept:
                beta0 = [1.0, 1.0]
            else:
                beta0 = [1.0, 1.0, 0.0]
            
            # Create ODR object and fit
            myodr = odr.ODR(mydata, linear, beta0=beta0)
            model = myodr.run()
            
            # Create predictions
            predicted = f(model.beta, X)
            
            # Calculate R²
            r2 = r2_score(y, predicted)
            
            result = {
                'predicted': predicted,
                'r2': r2,
                'slope': model.beta[0],  # First coefficient
                'model': model,
                'coef': model.beta[:2]  # First two coefficients for P and H
            }
            
            if verbose:
                print(f"R² Score: {r2:.4f}")
                print(f"Coefficients: {model.beta}")
            
            return result

        # Fit model (for non-ODR methods)
        if reg_type.lower() != 'odr':
            model.fit(X, y)
            predicted = model.predict(X)
            r2 = r2_score(y, predicted)
            
            # Get coefficients based on model type
            if reg_type.lower() == 'ransac':
                coef = model.estimator_.coef_
                intercept = model.estimator_.intercept_ if not zero_intercept else 0
            else:
                coef = model.coef_
                intercept = model.intercept_ if not zero_intercept else 0
            
            result = {
                'predicted': predicted,
                'r2': r2,
                'slope': coef[0],  # First coefficient
                'model': model,
                'coef': coef
            }
            
            if verbose:
                print(f"R² Score: {r2:.4f}")
                print(f"Coefficients: {coef}")
                if not zero_intercept:
                    print(f"Intercept: {intercept}")
            
            return result

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_coherence(self, coherence_dict: Dict, fmin: Optional[float] = None,
                      fmax: Optional[float] = None, figsize: Tuple = (10,6), out: bool = False) -> Optional[plt.Figure]:
        """
        Plot coherence results between barometer and seismometer/rotation components.
        
        Args:
            coherence_dict: Dictionary from _compute_coherence()
            fmin: Minimum frequency to plot
            fmax: Maximum frequency to plot
            figsize: Figure size tuple
            out: If True, return the figure object
        """
        return plot_coherence(coherence_dict, fmin, fmax, figsize, out)

    def plot_cross_correlations(self, results: Dict) -> None:
        """Plot cross-correlation results."""
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
        if len(results) == 1:
            axes = [axes]

        for ax, (channel, data) in zip(axes, results.items()):
            im = ax.pcolormesh(data['times'], data['lags'], data['ccf'].T, 
                             cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f'Cross-correlation: Barometer - {channel}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Lag (s)')
            plt.colorbar(im, ax=ax, label='CC Coefficient')

        plt.tight_layout()
        plt.show()

    def plot_cwt(self, results: Dict) -> None:
        """Plot continuous wavelet transform results."""
        n_plots = len(results)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        if n_plots == 1:
            axes = [axes]

        for ax, (channel, data) in zip(axes, results.items()):
            im = ax.pcolormesh(data['times'], data['frequencies'], 
                             data['cwt_power'], cmap='viridis')
            ax.set_title(f'CWT: {channel}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=ax, label='Power')

        plt.tight_layout()
        plt.show()

    def plot_waveforms(self, out: bool = False, time_unit: str = "hours", channel_type: str = "J") -> Optional[plt.Figure]:
        """
        Plot waveforms of rotation and pressure data.
        
        Args:
            out: Return figure handle if True
            time_unit: Time unit for x-axis ('hours', 'days', 'minutes', 'seconds')
            channel_type: Channel type to plot ('J' for rotation rate, 'A' for tilt, 'H' for acceleration)
        
        Returns:
            matplotlib.figure.Figure if out=True
        """
        if not hasattr(self, 'st'):
            raise ValueError("No data loaded. Run load_data() first")
            
        return plot_waveforms(self.st.copy(), self.config, out, time_unit, channel_type)

    def compare_prediction_methods(self, method1: Dict, method2: Dict, channel_type: str = "J", time_unit: str = "hours") -> None:
        """
        Compare two different prediction methods for all components.
        
        Args:
            method1: Dictionary describing first method, e.g.:
                    {'name': 'least_squares', 'reg_type': None, 'color': 'red'}
                    or {'name': 'regression', 'reg_type': 'ransac', 'color': 'blue'}
            method2: Dictionary describing second method
            channel_type: Type of channel to analyze ('J' for rotation rate or 'A' for tilt)
            time_unit: Time unit for x-axis ('minutes', 'hours', 'days', 'seconds')
        
        Example:
            # Compare least squares vs RANSAC regression
            bs.compare_prediction_methods(
                method1={'name': 'least_squares', 'reg_type': None, 'color': 'red'},
                method2={'name': 'regression', 'reg_type': 'ransac', 'color': 'blue'},
                channel_type='J'
            )
            
            # Compare two regression methods
            bs.compare_prediction_methods(
                method1={'name': 'regression', 'reg_type': 'ransac', 'color': 'blue'},
                method2={'name': 'regression', 'reg_type': 'theilsen', 'color': 'green'},
                channel_type='J'
            )
        """
        if not hasattr(self, 'st'):
            raise ValueError("No data loaded. Run load_data() first")
        
        # Validate methods
        for method in [method1, method2]:
            if 'name' not in method or 'color' not in method:
                raise ValueError("Each method must have 'name' and 'color' keys")
            if method['name'] not in ['least_squares', 'regression']:
                raise ValueError("Method name must be 'least_squares' or 'regression'")
            if method['name'] == 'regression' and 'reg_type' not in method:
                raise ValueError("Regression methods must specify 'reg_type'")
        
        # Set units and scaling
        if channel_type == 'J':
            ylabel = "Rotation Rate (nrad/s)"
            yscale = 1e9
            coef_unit = "nrad/s/hPa"
        else:
            ylabel = "Tilt (nrad)"
            yscale = 1e9
            coef_unit = "nrad/hPa"
        
        # Set time scaling
        tscale_dict = {
            "hours": 1/3600,
            "days": 1/86400,
            "minutes": 1/60,
            "seconds": 1
        }
        tscale = tscale_dict.get(time_unit, 1/60)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        font = 12
        
        # Store original predictions
        original_predictions = {}
        for comp in ['N', 'E', 'Z']:
            tr_pred = self.st.select(location="PP", channel=f"*{channel_type}{comp}").copy()
            if tr_pred:
                original_predictions[comp] = tr_pred[0].data
        
        # Format method labels
        methods = {
            'method1': {
                'color': method1['color'],
                'label': ('Least Squares' if method1['name'] == 'least_squares' 
                         else f"Regression ({method1['reg_type'].upper()})"),
                'name': method1['name'],
                'reg_type': method1.get('reg_type')
            },
            'method2': {
                'color': method2['color'],
                'label': ('Least Squares' if method2['name'] == 'least_squares' 
                         else f"Regression ({method2['reg_type'].upper()})"),
                'name': method2['name'],
                'reg_type': method2.get('reg_type')
            }
        }
        
        for i, comp in enumerate(['N', 'E', 'Z']):
            try:
                # Get original data
                tr_rot = self.st.select(channel=f"*{channel_type}{comp}").copy()[0]
                times = tr_rot.times(reftime=self.config['tbeg'])*tscale
                rot_data = tr_rot.data * yscale
                
                # Plot original data
                axes[i].plot(times, rot_data, 'k-', label=f'{comp}-component', alpha=0.7)
                
                # Try each method
                for method_key, props in methods.items():
                    # Make prediction
                    self.predict_tilt_from_pressure(
                        method=props['name'],
                        reg_type=props['reg_type'],
                        channel_type=channel_type,
                        verbose=False
                    )
                    
                    # Get prediction
                    tr_pred = self.st.select(location="PP", channel=f"*{channel_type}{comp}").copy()[0]
                    pred_data = tr_pred.data * yscale
                    
                    # Calculate variance reduction
                    var_red = self.variance_reduction(rot_data, rot_data - pred_data)
                    
                    # Get coefficients
                    p_coef = self.p_coefficient[comp] * yscale *1e2
                    h_coef = self.h_coefficient[comp] * yscale *1e2
                    
                    # Plot prediction
                    axes[i].plot(times, pred_data, color=props['color'],
                               label=f"{props['label']}\n(VR={var_red:.1f}%, P={p_coef:.2f}, H={h_coef:.2f})",
                               alpha=0.8)
                
                # Format subplot
                axes[i].set_ylabel(ylabel)
                axes[i].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                axes[i].grid(True, alpha=0.3)
                axes[i].spines[['right', 'top']].set_visible(False)
                
            except Exception as e:
                print(f"Could not process component {comp}: {str(e)}")
        
        # Set title and labels
        title = (f"{self.config['tbeg'].date} {str(self.config['tbeg'].time).split('.')[0]} - "
                f"{str(self.config['tend'].time).split('.')[0]} UTC")
        if 'fmin' in self.config and 'fmax' in self.config:
            title += f"\nf = {self.config['fmin']*1e3:.1f} - {self.config['fmax']*1e3:.1f} mHz"
        fig.suptitle(title, fontsize=font+2)
        
        axes[-1].set_xlabel(f"Time ({time_unit})")
        
        plt.tight_layout()
        plt.show()
        
        # Restore original predictions if they existed
        for comp in ['N', 'E', 'Z']:
            if comp in original_predictions:
                tr_pred = self.st.select(location="PP", channel=f"*{channel_type}{comp}").copy()[0]
                tr_pred.data = original_predictions[comp]

    def compare_spectra(self, method='fft', channel_type='J', fmin=0.0005, fmax=0.1, 
                    window='hann', log_scale=True, db_scale=False, figsize=(12, 10), compare_residual=True):
        """
        Compare spectra between observed data and either predicted data or residuals.
        
        Parameters:
        -----------
        method : str
            Method to compute spectra: 'fft', 'welch'
        channel_type : str
            Channel type to use: 
            'J' for rotation rate
            'A' for tilt
            'H' for acceleration
        fmin : float
            Minimum frequency to display
        fmax : float
            Maximum frequency to display
        window : str
            Window function for FFT ('hann', 'hamming', 'blackman', etc.)
        log_scale : bool
            Whether to use logarithmic scale for frequency axis
        db_scale : bool
            Whether to show amplitudes in decibels (dB). If True, computes 20*log10(amplitude)
        figsize : tuple
            Figure size
        compare_residual : bool
            If True, compare observed vs residual (observed-predicted)
            If False, compare observed vs predicted
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with the spectra comparison
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy import signal
        
        font = 12

        # Get locations from seis seeds as dict with component as key
        locations = {}
        for seed in self.config['seis_seeds']:
            net, sta, loc, cha = seed.split(".")
            locations[cha[-1]] = loc
        
        # Select the appropriate channels
        obs_N = self.st.select(location=locations['N'], channel=f'*{channel_type}N')[0].data
        obs_E = self.st.select(location=locations['E'], channel=f'*{channel_type}E')[0].data
        obs_Z = self.st.select(location=locations['Z'], channel=f'*{channel_type}Z')[0].data
        
        pred_N = self.st.select(location="PP", channel=f'*{channel_type}N')[0].data
        pred_E = self.st.select(location="PP", channel=f'*{channel_type}E')[0].data
        pred_Z = self.st.select(location="PP", channel=f'*{channel_type}Z')[0].data
        
        # Calculate residuals if needed
        if compare_residual:
            comp_N = obs_N - pred_N  # Residual
            comp_E = obs_E - pred_E
            comp_Z = obs_Z - pred_Z
            comp_label = "Residual"
        else:
            comp_N = pred_N  # Predicted
            comp_E = pred_E
            comp_Z = pred_Z
            comp_label = "Predicted"
        
        # Get sampling rate
        dt = self.st.select(component='N')[0].stats.delta
        
        # Compute spectra based on selected method
        if method == 'fft':
            def compute_spectrum(data):
                # Apply window
                windowed_data = data * signal.get_window(window, len(data))
                # Compute FFT
                fft = np.fft.rfft(windowed_data)
                # Get frequencies
                freqs = np.fft.rfftfreq(len(data), dt)
                # Compute amplitude spectrum
                amplitude = np.abs(fft) * 2.0 / len(data)
                return freqs, amplitude
        
        elif method == 'welch':
            def compute_spectrum(data):
                # Compute Welch PSD
                nperseg = min(8192, len(data)//8)
                freqs, psd = signal.welch(data, 1/dt, window=window, nperseg=nperseg)
                return freqs, np.sqrt(psd)  # Return amplitude spectrum
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'fft', 'welch'")
        
        # Compute spectra
        f_N, spec_obs_N = compute_spectrum(obs_N)
        f_E, spec_obs_E = compute_spectrum(obs_E)
        f_Z, spec_obs_Z = compute_spectrum(obs_Z)
        
        f_N, spec_comp_N = compute_spectrum(comp_N)
        f_E, spec_comp_E = compute_spectrum(comp_E)
        f_Z, spec_comp_Z = compute_spectrum(comp_Z)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, figure=fig, hspace=0.2)
        
        # Component labels and colors
        components = ['N', 'E', 'Z']
        colors = ['tab:blue', 'tab:red', 'tab:green']
        
        # Unit label based on channel type
        if channel_type == 'J':
            unit = 'rad/s'
        elif channel_type == 'A':
            unit = 'rad'
        elif channel_type == 'H':
            unit = 'm/s²'
        else:
            unit = 'units'
        
        # Plot spectra
        for i, (comp, color) in enumerate(zip(components, colors)):
            ax = fig.add_subplot(gs[i])
            
            # Get data for this component
            if comp == 'N':
                f, spec_obs = f_N, spec_obs_N
                spec_comp = spec_comp_N
            elif comp == 'E':
                f, spec_obs = f_E, spec_obs_E
                spec_comp = spec_comp_E
            else:  # Z
                f, spec_obs = f_Z, spec_obs_Z
                spec_comp = spec_comp_Z
            
            # Mask for frequency range
            mask = (f >= fmin) & (f <= fmax)
            
            # Convert to dB if requested
            if db_scale:
                if method == 'fft':
                    spec_obs_plot = 20 * np.log10(spec_obs[mask])
                    spec_comp_plot = 20 * np.log10(spec_comp[mask])
                elif method == 'welch':
                    spec_obs_plot = 10 * np.log10(spec_obs[mask])
                    spec_comp_plot = 10 * np.log10(spec_comp[mask])
            else:
                spec_obs_plot = spec_obs[mask]
                spec_comp_plot = spec_comp[mask]
            
            # Plot observed and comparison spectra
            ax.plot(f[mask], spec_obs_plot, color=color, label=f'Observed {comp}')
            ax.plot(f[mask], spec_comp_plot, color=color, linestyle='--', 
                    alpha=0.8, label=f'{comp_label} {comp}')
            
            # Fill between to highlight differences
            ax.fill_between(f[mask], spec_comp_plot, spec_obs_plot, 
                            color=color, alpha=0.2)
            
            # Set scales
            if log_scale:
                ax.set_xscale('log')
                if not db_scale:  # Only use log scale for y-axis if not in dB
                    ax.set_yscale('log')
            
            # Set labels and title
            if db_scale:
                ax.set_ylabel(f'Amplitude (dB rel. 1 {unit})', fontsize=font)
            else:
                if method == 'welch':
                    ax.set_ylabel(f'Power Spectral Density (({unit})^2/Hz)', fontsize=font)
                else:
                    ax.set_ylabel(f'Amplitude ({unit}/√Hz)')
            if i == 2:  # Only add x-label to bottom plot
                ax.set_xlabel('Frequency (Hz)', fontsize=font)
            
            # Add component label
            ax.text(0.02, 0.98, f'Component {comp}', transform=ax.transAxes,
                    fontsize=font+1, fontweight='bold')
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add grid
            ax.grid(True, which='both', alpha=0.3)
        
        # Add title
        channel_name = {
            'J': 'Rotation Rate',
            'A': 'Tilt',
            'H': 'Acceleration'
        }[channel_type]
        
        title = f"{channel_name} Spectra Comparison using {str(method).upper()}\n"
        title += f"Observed vs {'Residual' if compare_residual else 'Predicted'} | "
        title += f"Frequency range: {fmin*1e3:.1f} - {fmax*1e3:.1f} mHz"
        fig.suptitle(title, fontsize=font+1, y=0.95)
        
        # Calculate variance reduction for each component
        variance_reductions = {}
        for comp in components:
            try:
                obs_data = self.st.select(channel=f'*{channel_type}{comp}')[0].data
                pred_data = self.st.select(location="PP", channel=f'*{channel_type}{comp}')[0].data
                residual = obs_data - pred_data
                var_red = self.variance_reduction(obs_data, residual)
                variance_reductions[comp] = var_red
            except Exception as e:
                print(f"Could not calculate variance reduction for component {comp}: {str(e)}")
                variance_reductions[comp] = None
        
        # Add variance reduction if available
        vr_text = "Variance Reduction: "
        has_vr = False
        for comp in components:
            if variance_reductions.get(comp) is not None:
                vr_text += f"{comp}: {variance_reductions[comp]:.0f}%, "
                has_vr = True
        
        if has_vr:
            fig.text(0.5, 0.01, vr_text[:-2], ha='center', fontsize=12)
        
        plt.tight_layout()
        return fig

    def plot_residuals(self, time_unit: str = "minutes", channel_type: str = "J", out: bool = False) -> Optional[plt.Figure]:
        """
        Plot residuals between observed and predicted data.
        
        Args:
            time_unit: Time unit for x-axis ('minutes', 'hours', 'days', 'seconds')
            channel_type: Type of channel to plot:
                         'J' for rotation rate
                         'A' for tilt
                         'H' for acceleration
            out: If True, return figure object
        """
        if not hasattr(self, 'st'):
            raise ValueError("No data loaded. Run load_data() first")
        
        if channel_type not in ['J', 'A', 'H']:
            raise ValueError("channel_type must be 'J', 'A', or 'H'")
            
        # Create config dict with necessary values
        plot_config = self.config.copy()
        if hasattr(self, 'p_coefficient'):
            plot_config['p_coefficient'] = self.p_coefficient
        if hasattr(self, 'h_coefficient'):
            plot_config['h_coefficient'] = self.h_coefficient
            
        return plot_residuals(self.st.copy(), plot_config, time_unit, channel_type, out)

    def plot_residuals_derivatives(self, time_unit: str = "minutes", channel_type: str = "J", 
                                  out: bool = False) -> Optional[plt.Figure]:
        """
        Plot residuals between observed and predicted data using model_tilt_from_pressure output.
        
        This method creates a waveform plot similar to plot_residuals but uses the model
        output from model_tilt_from_pressure which includes derivatives of pressure and
        Hilbert transform.
        
        Args:
            time_unit: Time unit for x-axis ('minutes', 'hours', 'days', 'seconds')
            channel_type: Type of channel to plot:
                         'J' for rotation rate
                         'A' for tilt
                         'H' for acceleration
            out: If True, return figure object
        
        Returns:
            matplotlib.figure.Figure if out=True, None otherwise
            
        Raises:
            ValueError: If no data is loaded or coefficients are not available
        """
        if not hasattr(self, 'st'):
            raise ValueError("No data loaded. Run load_data() first")
        
        if channel_type not in ['J', 'A', 'H']:
            raise ValueError("channel_type must be 'J', 'A', or 'H'")
        
        # Check if coefficients are available
        if not hasattr(self, 'p_coefficient') or not hasattr(self, 'h_coefficient'):
            raise ValueError("Coefficients not available. Run model_tilt_from_pressure() first")
        
        # Create config dict with coefficient data
        plot_config = self.config.copy()
        plot_config['p_coefficient'] = self.p_coefficient
        plot_config['h_coefficient'] = self.h_coefficient
        
        # Add derivative coefficients if available
        if hasattr(self, 'dp_coefficient') and hasattr(self, 'dh_coefficient'):
            plot_config['dp_coefficient'] = self.dp_coefficient
            plot_config['dh_coefficient'] = self.dh_coefficient
        
        return plot_residuals_derivatives(self.st.copy(), plot_config, time_unit, channel_type, out)

    def plot_scatter_correlations(self, time_unit: str = "minutes", channel_type: str = "A", 
                                 out: bool = False, figsize: Tuple[int, int] = (16, 12),
                                 alpha: float = 0.6, s: float = 1.0) -> Optional[plt.Figure]:
        """
        Plot scatter plots of seismic data (Z, N, E) vs pressure data (p, h, dh, dp).
        
        Creates a 3x4 grid showing correlations between seismic components and pressure data.
        
        Args:
            time_unit: Time unit for x-axis ('minutes', 'hours', 'days', 'seconds')
            channel_type: Type of channel to plot ('J', 'A', 'H')
            out: If True, return figure object
            figsize: Figure size (width, height)
            alpha: Transparency for scatter points
            s: Size of scatter points
            
        Returns:
            matplotlib Figure object if out=True, otherwise None
        """
        if len(self.st) == 0:
            raise ValueError("No data loaded. Run load_data() first")
        
        if channel_type not in ['J', 'A', 'H']:
            raise ValueError("channel_type must be 'J', 'A', or 'H'")
        
        return plot_scatter_correlations(self, time_unit, channel_type, out, figsize, alpha, s)

