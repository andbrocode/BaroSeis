import numpy as np
import os

from typing import List, Dict, Optional, Union, Tuple
from obspy import Stream, UTCDateTime, Inventory
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.clients.fdsn import Client as FDSNClient


class baroArray:
    """
    Class for handling an array of barometer stations.
    
    Attributes:
        st (Stream): Stream containing barometer data
        config (Dict): Configuration dictionary
        inventory (Inventory): Station metadata
    """

    def __init__(self, config: Dict = {}):
        """
        Initialize baroArray object.
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        default_config = {
            # Time parameters
            'tbeg': None,
            'tend': None,
            
            # Data source
            'data_source': 'sds',  # 'sds' or 'fdsn'
            'sds_path': './data',  # Path to SDS archive
            'fdsn_server': 'IRIS', # FDSN server to use
            
            # Station parameters
            'seeds': [],  # List of SEED IDs for barometer stations
            'reference_station': None,  # Reference station for coherence
            
            # Processing parameters
            'fmin': None,  # Minimum frequency in Hz  
            'fmax': None,  # Maximum frequency in Hz
            'sampling_rate': 1.0,  # Target sampling rate
            
            # Other parameters
            'verbose': False,
            'merge_nans': True,
            'inventory': Inventory(),
        }
        
        # Update default config with provided config
        self.config = {**default_config, **config}
        
        # Initialize empty stream and inventory
        self.st = Stream()
        self.st0 = None  # For storing raw data
        self.inventory = Inventory()

        # Convert time strings to UTCDateTime if provided
        for key in ['tbeg', 'tend']:
            if self.config[key] is not None:
                self.config[key] = UTCDateTime(self.config[key])

    def load_data(self, 
                 tbeg: Optional[str] = None, 
                 tend: Optional[str] = None,
                 data_source: Optional[str] = None) -> None:
        """
        Load barometer data from SDS archive or FDSN service.
        
        Args:
            tbeg: Start time (overrides config)
            tend: End time (overrides config) 
            data_source: Data source type ("sds" or "fdsn")
        """
        # Update times if provided
        if tbeg is not None:
            self.config['tbeg'] = UTCDateTime(tbeg)
        if tend is not None:
            self.config['tend'] = UTCDateTime(tend)

        # Update data source if provided
        if data_source is not None:
            self.config['data_source'] = data_source
            
        # Check required parameters
        if self.config['tbeg'] is None or self.config['tend'] is None:
            raise ValueError("Start and end times must be specified")
            
        if not self.config['seeds']:
            raise ValueError("No station seeds specified")
        
        # add time buffer to time window
        self.config['t1'] = self.config['tbeg'] - self.config['time_buffer']
        self.config['t2'] = self.config['tend'] + self.config['time_buffer']

        # Load data based on source type
        if self.config['data_source'].lower() == "sds":
            self._load_from_sds()
        elif self.config['data_source'].lower() == "fdsn":
            self._load_from_fdsn()
        else:
            raise ValueError("data_source must be 'sds' or 'fdsn'")

        # Merge traces if needed
        self.st.merge(method=1)
        
        # Trim data to requested time window
        self.st.trim(self.config['tbeg'], self.config['tend'])

        # Check data quality
        self._check_data_quality()

        # Store raw data
        self.st0 = self.st.copy()

    def _check_data_quality(self) -> None:
        """Check data quality."""
        # Check for gaps
        gaps = self.st.get_gaps()
        if len(gaps) > 0:
            print(f"Warning: {len(gaps)} gaps found in data")
        
        # Get expected number of samples
        npts_expected = int((self.config['tend'] - self.config['tbeg']) * self.config['sampling_rate'])

        # Check for nans and merge if needed
        npts_differ = False
        for tr in self.st:
            if tr.stats.npts != npts_expected:
                npts_differ = True
                print(f"Warning: {tr.id} has {tr.stats.npts} samples, but {npts_expected} samples expected")
            
            # Check for all NaN values
            if np.isnan(tr.data).all():
                print(f"Warning: {tr.id} has only NaN values! Replacing with zeros.")
                tr.data = np.zeros_like(tr.data)
            # Check for some NaN values
            elif np.isnan(tr.data).any():
                print(f"Warning: {tr.id} has some NaN values")
                if self.config['merge_nans']:
                    if self.config['verbose']:
                        print(f"Warning: Interpolating NaN values for {tr.id}")
                    tr.data = self._interpolate_nan(tr.data)

        if npts_differ:
            self._trim_common_channels()

    def _trim_common_channels(self) -> None:
        """Trim all traces to the shortest trace."""
        npts_min = min([tr.stats.npts for tr in self.st])
        for tr in self.st:
            tr.trim(self.config['tbeg'], self.config['tbeg'] + npts_min/self.config['sampling_rate'])

    def _interpolate_nan(self, array_like):

        from numpy import isnan, interp

        array = array_like.copy()

        nans = isnan(array)

        def get_x(a):
            return a.nonzero()[0]

        array[nans] = interp(get_x(nans), get_x(~nans), array[~nans])

        return array

    def _load_from_sds(self) -> None:
        """
        Load data from local SDS archive using parallel processing.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from obspy import read_inventory

        def _load_seed(seed_info):
            """Helper function to load a single seed and its inventory."""
            path, seed = seed_info
            net, sta, loc, cha = seed.split(".")
            result = {'stream': Stream(), 'inventory': None}

            try:
                # Initialize SDS client
                client = SDSClient(path)
                
                # Get waveforms
                st = client.get_waveforms(
                    network=net, station=sta, location=loc, channel=cha,
                    starttime=self.config['t1'],
                    endtime=self.config['t2']
                )

                if self.config['verbose']:
                    if len(st) > 0:
                        print(f"Loaded {seed} from SDS archive")
                    else:
                        print(f"No data found for {seed}")

                result['stream'] = st

            except Exception as e:
                print(f"Failed to load {seed}: {str(e)}")

            try:
                if self.config.get('inventory_path') is not None:
                    inv = read_inventory(self.config['inventory_path']+f"station_{net}_{sta}.xml")
                    result['inventory'] = inv
                else:
                    if self.config['verbose']:
                        print(f"No inventory path specified (config['inventory_path']), skipping inventory loading for {seed}")

            except Exception as e:
                print(f"Failed to load inventory for {seed}: {str(e)}")

            return result

        # Prepare all seeds to load
        seeds_to_load = [(self.config['sds_path'], seed) for seed in self.config['seeds']]

        # Load all data in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_load_seed, seed_info) for seed_info in seeds_to_load]
            for future in as_completed(futures):
                result = future.result()
                if result['stream']:
                    self.st += result['stream']
                if result['inventory']:
                    self.inventory += result['inventory']

    def _load_from_fdsn(self) -> None:
        """Load data from FDSN web service."""
        client = FDSNClient(self.config['fdsn_server'])
        
        for seed in self.config['seeds']:
            net, sta, loc, cha = seed.split(".")
            try:
                st = client.get_waveforms(
                    network=net, station=sta, location=loc, channel=cha,
                    starttime=self.config['t1'], 
                    endtime=self.config['t2']
                )
                
                if self.config['verbose']:
                    if len(st) > 0:
                        print(f"Loaded {seed} from {self.config['fdsn_server']}")
                    else:
                        print(f"No data found for {seed}")

                self.st += st
                    
            except Exception as e:
                print(f"Failed to load {seed}: {str(e)}")

            try:
                inv = client.get_stations(
                    network=net, station=sta, location=loc, channel=cha,
                    starttime=self.config['t1'],
                    endtime=self.config['t2']
                )

                self.inventory += inv

            except Exception as e:
                print(f"Failed to load inventory for {seed}: {str(e)}")

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
            # Write each trace to SDS structure
            for tr in stream:
                # Get time info
                year = str(self.config['tbeg'].year)
                julday = "%03d" % self.config['tbeg'].julday
                
                # Create filename
                filename = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}.D.{year}.{julday}"
                
                # Create directory structure
                net_dir = os.path.join(sds_path, year, tr.stats.network)
                sta_dir = os.path.join(net_dir, tr.stats.station)
                cha_dir = os.path.join(sta_dir, f"{tr.stats.channel}.D")
                
                # Create directories if they don't exist
                for directory in [net_dir, sta_dir, cha_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                
                # Write trace
                tr_stream = Stream([tr])
                tr_stream.write(os.path.join(cha_dir, filename), format="MSEED")
                
                if self.config['verbose']:
                    print(f"Written: {filename}")
                
        except Exception as e:
            print(f"Error writing to SDS: {str(e)}")

    @staticmethod
    def get_time_intervals(tbeg, tend, interval_seconds, interval_overlap):

        from obspy import UTCDateTime

        tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

        times = []
        t1, t2 = tbeg, tbeg + interval_seconds

        while t2 <= tend:
            times.append((t1, t2))
            t1 = t1 + interval_seconds - interval_overlap
            t2 = t2 + interval_seconds - interval_overlap

        return times

    def filter_data(self, fmin: Optional[float] = None, fmax: Optional[float] = None) -> None:
        """
        Apply bandpass filter to data streams.
        
        Args:
            fmin: Minimum frequency in Hz (overrides config)
            fmax: Maximum frequency in Hz (overrides config)
        """
        if len(self.st) == 0:
            return
        
        # Reset data with raw data
        self.st = self.st0.copy()

        # Update fmin and fmax if provided
        if fmin is not None:
            self.config['fmin'] = fmin
        if fmax is not None:
            self.config['fmax'] = fmax

        # Detrend and taper data
        if self.config['verbose']:
            print("Detrending and tapering data...")
        self.st.detrend('linear')
        self.st.detrend('simple')
        self.st.detrend('demean')
        self.st.taper(0.05, type='cosine')

        # Apply bandpass filter if frequencies specified
        if self.config['fmin'] is not None and self.config['fmax'] is not None:
            if self.config['verbose']:
                print(f"Applying bandpass filter {self.config['fmin']} - {self.config['fmax']} Hz...")
            self.st.filter(
                'bandpass',
                freqmin=self.config['fmin'],
                freqmax=self.config['fmax'],
                corners=4,
                zerophase=True
            )

        # Apply highpass filter if fmin specified
        elif self.config['fmin'] is not None:
            if self.config['verbose']:
                print(f"Applying highpass filter {self.config['fmin']} Hz...")
            self.st.filter(
                'highpass',
                freq=self.config['fmin'],
                corners=4,
                zerophase=True
            )

        # Apply lowpass filter if fmax specified
        elif self.config['fmax'] is not None:
            if self.config['verbose']:
                print(f"Applying lowpass filter {self.config['fmax']} Hz...")
            self.st.filter(
                'lowpass', 
                freq=self.config['fmax'],
                corners=4,
                zerophase=True
            )

    def compute_station_distances(self, output: str = "km", ref_station: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Compute distances between barometer stations.
        
        Args:
            output: Unit for distances ("km" or "m")
            ref_station: Optional reference station SEED ID
            
        Returns:
            Dictionary containing:
                - distances: Matrix/dict of station distances
                - coordinates: Station coordinates in local reference frame
                - reference: Reference station info
        """
        from obspy.geodetics.base import gps2dist_azimuth
        
        # Check inventory exists
        if not self.inventory:
            raise ValueError("No station inventory available")
        
        # Get station coordinates
        stations = {}
        for seed in self.config['seeds']:
            try:
                net, sta = seed.split(".")[:2]
                name = f"{net}.{sta}"
                try:
                    coords = self.inventory.get_coordinates(seed)
                except:
                    coords = self.inventory.get_coordinates(f"{net}.{sta}..HHZ")
                stations[name] = {
                    'seed': seed,
                    'lat': coords['latitude'],
                    'lon': coords['longitude'],
                    'elev': coords['elevation']
                }
            except Exception as e:
                print(f"Warning: Could not get coordinates for {seed}: {str(e)}")
                continue
            
        if not stations:
            raise ValueError("No valid station coordinates found")
        
        # Set reference station
        if ref_station:
            ref_net, ref_sta = ref_station.split(".")[:2]
            ref_name = f"{ref_net}.{ref_sta}"
            if ref_name not in stations:
                raise ValueError(f"Reference station {ref_station} not found in inventory")
            reference = stations[ref_name]
        elif self.config['reference_station']:
            ref_net, ref_sta = self.config['reference_station'].split(".")[:2]
            ref_name = f"{ref_net}.{ref_sta}"
            if ref_name not in stations:
                raise ValueError(f"Reference station {self.config['reference_station']} not found in inventory")
            reference = stations[ref_name]
        else:
            print(" -> no reference station specified!")
            return None
        
        # Compute distances and local coordinates
        distances = {}
        coordinates = {}
        
        # Add reference coordinates to output
        coordinates['reference'] = {
            'east': 0,
            'north': 0,
            'elevation': reference['elev']
        }
        
        for name, station in stations.items():
            # Skip if reference station
            if station == reference:
                continue
            
            # Compute distance and azimuth
            dist, az, _ = gps2dist_azimuth(
                reference['lat'], 
                reference['lon'],
                station['lat'], 
                station['lon']
            )
            
            # Convert to requested units
            if output == "km":
                dist /= 1000
            elif output != "m":
                raise ValueError("output must be 'km' or 'm'")
            
            # Store distance
            distances[name] = round(dist, 3)
            
            # Compute local coordinates
            coordinates[name] = {
                'east': dist * np.sin(np.deg2rad(az)),
                'north': dist * np.cos(np.deg2rad(az)),
                'elevation': station['elev']
            }
        
        if verbose:
            print("\nStation distances:")
            for name, dist in distances.items():
                unit = "km" if output == "km" else "m"
                print(f"{name}: {dist:.2f} {unit}")
            
        return {
            'distances': distances,
            'coordinates': coordinates,
            'reference': reference
        }

    def compute_baro_gradient(self, ref_station: Optional[str] = None, vp: float = 6.0, 
                             vs: float = 3.5, sigmau: float = 1e-8, keep_z: bool = False, mode: str = 'tilt') -> Optional[Stream]:
        """
        Compute barometer gradient using array data.
        
        Args:
            ref_station: Optional reference station SEED ID
            vp: P-wave velocity in km/s
            vs: S-wave velocity in km/s
            sigmau: Uncertainty threshold in barometric pressure amplitude
            keep_z: Include vertical component in output if True
            mode: 'tilt' or 'rotation'
        Returns:
            Stream containing gradient components (East, North, optional Z) or None if computation fails
        """
        from obspy.signal.array_analysis import array_rotation_strain
        
        # add parameters to config
        self.config['vp'] = vp
        self.config['vs'] = vs
        self.config['sigmau'] = sigmau

        if len(self.st) < 3:
            print("At least 3 stations required for gradient calculation")
            return None
        
        # Get station coordinates and convert to local reference frame
        results = self.compute_station_distances(ref_station=ref_station, verbose=False)
        if results is None:
            print("Could not compute station distances")
            return None
        
        # Initialize arrays
        dist = []
        tsz = []  # Vertical pressure data
        
        # Get reference coordinates
        ref_coords = results['coordinates']['reference']
        ref_name = f"{self.config['reference_station'].split('.')[0]}.{self.config['reference_station'].split('.')[1]}"
        
        # Collect station data and coordinates
        for tr in self.st:
            net, sta = tr.id.split(".")[:2]
            station_name = f"{net}.{sta}"
            
            # Get coordinates
            if station_name in results['coordinates']:
                coords = results['coordinates'][station_name]
                dist.append([
                    coords['east'],
                    coords['north'],
                    coords['elevation'] - ref_coords['elevation']
                ])
                
                # Add pressure data if barometer channel
                if "D" in tr.stats.channel:
                    tsz.append(tr.data)
                
            elif station_name == ref_name:
                dist.append([
                    ref_coords['east'],
                    ref_coords['north'],
                    0.0  # Reference elevation difference is 0
                ])
                if "D" in tr.stats.channel:
                    tsz.append(tr.data)
            else:
                print(f"Warning: No coordinates found for {station_name}")
            
        # Convert to numpy arrays
        dist = np.array(dist)
        tsz = np.array(tsz)
        
        if len(tsz) == 0:
            print("No barometer data found")
            return None
        
        # Pressure has no horizontal components
        tse = np.zeros_like(tsz)
        tsn = np.zeros_like(tsz)
        
        # Define array for subarray stations
        substations = np.arange(len(tsz))
        
        try:
            # Compute spatial derivatives
            gradient = array_rotation_strain(
                substations,
                np.transpose(tse),
                np.transpose(tsn),
                np.transpose(tsz),
                vp, vs, dist, sigmau
            )
        except Exception as e:
            print(f"Array rotation strain calculation failed: {str(e)}")
            return None
        
        # Create output stream
        ref_net, ref_sta, ref_loc, ref_cha = self.config['reference_station'].split(".")
        out = Stream()
        
        # Add components to output stream
        components = ['Z', 'N', 'E'] if keep_z else ['N', 'E']
        gradient_keys = ['ts_w3', 'ts_w2', 'ts_w1'] if keep_z else ['ts_w2', 'ts_w1']
        
        for comp, key in zip(components, gradient_keys):
            tr = self.st.select(network=ref_net, station=ref_sta, 
                               location=ref_loc, channel=ref_cha)[0].copy()
            tr.stats.channel = tr.stats.channel[:-1] + comp
            tr.data = gradient[key]
            out += tr
        
        if mode == 'tilt':
            tilt_e = out.select(channel='*N')[0].data
            tilt_n = out.select(channel='*E')[0].data*-1
            out.select(channel='*N')[0].data = tilt_n
            out.select(channel='*E')[0].data = tilt_e

        # Detrend output
        out = out.detrend('linear')
        
        # Store gradient stream
        self.st_grad = out.copy()
        
        # Add gradient magnitude
        self.gradient_magnitude = self.get_gradient_magnitude(out.select(channel='*N')[0].data, out.select(channel='*E')[0].data)

        # Add gradient angle
        self.gradient_angle_deg = self.get_azimuth(out.select(channel='*N')[0].data, out.select(channel='*E')[0].data, unit='degree')
        self.gradient_angle_rad = self.get_azimuth(out.select(channel='*N')[0].data, out.select(channel='*E')[0].data, unit='radian')

        if self.config['verbose']:
            print("\nComputed pressure gradients:")
            print(self.st_grad)
        
        return self.st_grad

    @staticmethod
    def get_azimuth(north: Union[float, np.ndarray], 
                    east: Union[float, np.ndarray], 
                    unit: str = 'degree') -> Union[float, np.ndarray]:
        """
        Calculate azimuth angle from north and east components.
        
        Azimuth is measured from North (0°) going clockwise:
        - North: 0° (or 0 rad)
        - East: 90° (or π/2 rad)
        - South: 180° (or π rad)
        - West: 270° (or 3π/2 rad)
        
        Args:
            north: North component (can be scalar or array)
            east: East component (can be scalar or array)
            unit: Output unit, either 'degree' or 'radian' (default: 'degree')
        
        Returns:
            Azimuth angle in specified unit. For degrees: 0-360°, for radians: 0-2π
        
        Example:
            >>> azimuth_deg = baroArray.get_azimuth(1.0, 0.0, unit='degree')  # Returns 0.0 (North)
            >>> azimuth_deg = baroArray.get_azimuth(0.0, 1.0, unit='degree')  # Returns 90.0 (East)
            >>> azimuth_rad = baroArray.get_azimuth(1.0, 1.0, unit='radian')  # Returns π/4
        """
        # Check if inputs are scalars
        is_scalar = np.isscalar(north) and np.isscalar(east)
        
        # Convert to numpy arrays if needed
        north = np.asarray(north)
        east = np.asarray(east)
        
        # Calculate azimuth: arctan2(east, north) gives angle from north going clockwise
        azimuth_rad = np.arctan2(east, north)
        
        # Convert to 0-2π range (instead of -π to π)
        azimuth_rad = np.mod(azimuth_rad + 2 * np.pi, 2 * np.pi)
        
        # Convert to degrees if requested
        if unit.lower() in ['degree', 'deg', 'degrees']:
            azimuth = np.rad2deg(azimuth_rad)
        elif unit.lower() in ['radian', 'rad', 'radians']:
            azimuth = azimuth_rad
        else:
            raise ValueError(f"Unknown unit: {unit}. Use 'degree' or 'radian'")
        
        # Return scalar if input was scalar
        if is_scalar:
            return float(azimuth)
        
        return azimuth

    @staticmethod
    def get_gradient_magnitude(north: Union[float, np.ndarray], 
                               east: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate gradient magnitude from north and east components.
        """
        return np.sqrt(north**2 + east**2)

    def scale_data(self, scale: float = 1.0) -> None:
        """
        Scale data by a factor.
        """
        self.config['scale'] = scale
        
        st_new = self.st.copy()
        for tr in st_new:
            tr.data = tr.data * scale
        self.st = st_new

    def compute_coherence(self, ref_station: Optional[str] = None, 
                         window_sec: float = 3600.0, overlap: float = 0.5,
                         smooth_points: int = 31) -> Dict:
        """
        Compute coherence between array stations and reference station.
        
        Args:
            ref_station: Reference station SEED ID (uses config if None)
            window_sec: Window length in seconds for coherence calculation
            overlap: Window overlap fraction (0-1)
            smooth_points: Number of points for coherence smoothing
            
        Returns:
            Dictionary containing frequencies and coherence for each station
        """
        from scipy.signal import coherence
        from scipy.signal.windows import hann
        
        # Get reference station
        if ref_station is None:
            if self.config.get('reference_station') is None:
                raise ValueError("No reference station specified")
            ref_station = self.config['reference_station']
        
        # Get reference trace
        ref_net, ref_sta, ref_loc, ref_cha = ref_station.split(".")
        ref_tr = self.st.select(network=ref_net, station=ref_sta, 
                               location=ref_loc, channel=ref_cha)
        
        if not ref_tr:
            raise ValueError(f"Reference station {ref_station} not found in data")
        ref_tr = ref_tr[0]
        
        # Calculate window parameters
        npts = int(window_sec * ref_tr.stats.sampling_rate)
        if npts > len(ref_tr.data):
            raise ValueError(f"Window length ({window_sec}s) longer than data duration")
        noverlap = int(npts * overlap)
        
        # Create output dictionary
        coherence_dict = {}
        
        # Compute coherence for each station
        for tr in self.st:
            if tr.id == ref_tr.id:
                continue
            
            # Check sampling rates match
            if tr.stats.sampling_rate != ref_tr.stats.sampling_rate:
                print(f"Warning: Sampling rate mismatch for {tr.id}, skipping")
                continue
            
            # Check for NaN values
            if np.isnan(tr.data).any() or np.isnan(ref_tr.data).any():
                print(f"Warning: NaN values found for {tr.id}, skipping")
                continue
            
            try:
                # Compute coherence
                freq, coh = coherence(
                    ref_tr.data, tr.data,
                    fs=tr.stats.sampling_rate,
                    nperseg=npts,
                    noverlap=noverlap,
                    window=hann(npts)
                )
                
                # Smooth coherence if requested
                if smooth_points > 0:
                    from scipy.signal import savgol_filter
                    if len(coh) > smooth_points:
                        coh = savgol_filter(coh, smooth_points, 3)
                    else:
                        print(f"Warning: Not enough points for smoothing {tr.id}")
                
                coherence_dict[tr.id] = {
                    'frequencies': freq,
                    'coherence': coh,
                    'window_sec': window_sec,
                    'overlap': overlap
                }
                
            except Exception as e:
                print(f"Error computing coherence for {tr.id}: {str(e)}")
                continue
        
        if not coherence_dict:
            raise ValueError("No valid coherence results computed")
        
        return coherence_dict

    def plot_coherence(self, coherence_dict: Dict, fmin: Optional[float] = None,
                      fmax: Optional[float] = None, figsize: Tuple = (10,6)) -> None:
        """
        Plot coherence results.
        
        Args:
            coherence_dict: Dictionary from compute_coherence()
            fmin: Minimum frequency to plot
            fmax: Maximum frequency to plot
            figsize: Figure size tuple
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for station_id, data in coherence_dict.items():
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
            ax.semilogx(freq, coh, label=station_id, alpha=0.7)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title('Station Coherence with Reference Station')
        ax.legend(loc='best')
        ax.set_ylim(0, 1.01)
        
        if fmin is not None:
            ax.set_xlim(left=fmin)
        if fmax is not None:
            ax.set_xlim(right=fmax)
        
        plt.tight_layout()
        plt.show()

    def plot_gradient(self, figsize: Tuple[int, int] = (15, 10), out: bool = False, unwrap: bool = False) -> None:
        """
        Plot pressure gradient analysis including:
        - All station pressures with reference station highlighted
        - Gradient components (E-W, N-S)
        - Gradient angle over time
        - Polar plot of gradient direction
        - Unwrap gradient angle if unwrap=True
        
        Args:
            figsize: Figure size tuple (width, height)
            out: Return figure handle if True
            unwrap: Unwrap gradient angle if True
        Returns:
            matplotlib.figure.Figure if out=True
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        if not hasattr(self, 'st_grad'):
            raise ValueError("No gradient data found. Run compute_baro_gradient() first.")
        
        # Get reference station pressure data
        ref_sta = self.config['reference_station']
        if not ref_sta:
            raise ValueError("No reference station specified in config")
        
        ref_pressure = self.st.select(id=ref_sta)[0]
        ref_name = ref_pressure.id
        
        # Get gradient components
        grad_e = self.st_grad.select(channel="*E")[0]
        grad_n = self.st_grad.select(channel="*N")[0]
        
        # scaling
        grad_e.data = grad_e.data * 1e3 # from Pa/m to Pa/km
        grad_n.data = grad_n.data * 1e3 # from Pa/m to Pa/km

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, width_ratios=[3, 2.5], height_ratios=[1, 1, 1], hspace=0.3, wspace=0.2)
        
        # Time series plots
        ax1 = fig.add_subplot(gs[0, 0])  # Pressure
        ax2 = fig.add_subplot(gs[1, 0])  # Gradients
        ax3 = fig.add_subplot(gs[2, 0])  # Gradient angle
        ax4 = fig.add_subplot(gs[:, 1], projection='polar')  # Polar plot
        
        # Plot all station pressures
        times = ref_pressure.times() / 3600  # Convert to hours
        
        # Plot other stations first in grey
        for tr in self.st:
            if tr.id != ref_sta and "D" in tr.stats.channel:
                ax1.plot(times, tr.data, 'grey', alpha=0.3, linewidth=1, zorder=5)
        
        # Plot reference station on top in black
        ax1.plot(times, ref_pressure.data, 'k-', label=ref_name, linewidth=1.5, zorder=5)
        ax1.set_ylabel('Pressure (Pa)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, zorder=0)
        
        # Plot gradients
        ax2.plot(times, grad_e.data, c='tab:blue', linewidth=1.5, label='East', alpha=0.7, zorder=5)
        ax2.plot(times, grad_n.data, c='tab:orange', linewidth=1.5, label='North', alpha=0.7, zorder=5)
        ax2.set_ylabel('Gradient (Pa/km)', fontsize=12)
        ax2.legend(ncol=2, loc='lower right')
        ax2.grid(True, alpha=0.3, zorder=0)
        
        # Plot gradient angle
        if unwrap:
            angles_deg = np.unwrap(self.gradient_angle_deg, period=360)
        else:
            angles_deg = self.gradient_angle_deg
    
        ax3.scatter(times, angles_deg, c=times, cmap='rainbow', alpha=0.5, s=2, zorder=5)
        ax3.set_ylabel('Gradient\nAzimuth (°)', fontsize=12)
        if unwrap:
            ticks  = np.arange(np.round(min(angles_deg/10),0)*10, np.round(max(angles_deg/10),0)*10, 90)
            ax3.set_yticks(ticks)
            ax3.set_yticklabels([f'{tick:.1f}' for tick in np.mod(ticks, 360)])
        else:
            ax3.set_yticks(np.arange(0, 361, 90))
            ax3.set_ylim(-5, 365)
        ax3.grid(True, alpha=0.3, zorder=0)
        ax3.set_xlabel('Time (hours)', fontsize=12)
        
        # Polar plot of gradients
        scatterdummy = ax4.scatter(
            self.gradient_angle_rad,
            self.gradient_magnitude/max(self.gradient_magnitude)*0,
            c=times, 
            cmap='rainbow', 
            alpha=1.0, 
            s=0.5,
            zorder=5
        )

        scatter = ax4.scatter(
            self.gradient_angle_rad,
            self.gradient_magnitude/max(self.gradient_magnitude),
            c=times, 
            cmap='rainbow', 
            alpha=0.5, 
            s=5,
            zorder=5
        )
        
        # add text with frequency band
        if hasattr(self, 'config') and 'fmin' in self.config and 'fmax' in self.config:
            ax1.text(0.8, 0.15, f'f = {self.config["fmin"]*1e3:.1f} - {self.config["fmax"]*1e3:.1f} mHz',
                     transform=ax1.transAxes, fontsize=11, ha='center', va='top')

        # Add colorbar
        cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(scatterdummy, cax=cax)
        cbar.set_label('Time (hours)', fontsize=12)
        
        # Customize polar plot
        ax4.set_theta_zero_location('N')
        ax4.set_theta_direction(-1)  # Clockwise
        # ax4.set_rlabel('Gradient Magnitude (Pa/km)', fontsize=12)
        ax4.set_title('Gradient Direction', fontsize=12)
        
        # Add labels for subplots
        ax1.text(0.01, 0.88, "(a)", fontsize=11, transform=ax1.transAxes)
        ax2.text(0.01, 0.88, "(b)", fontsize=11, transform=ax2.transAxes)
        ax3.text(0.01, 0.88, "(c)", fontsize=11, transform=ax3.transAxes)
        ax4.text(0.02, 0.96, "(d)", fontsize=11, transform=ax4.transAxes)

        # Add timestamp to plot
        start_time = ref_pressure.stats.starttime
        title = f'Spatial Pressure Gradients - {start_time.date} {str(start_time.time)[:8]} UTC'
        if hasattr(self, 'config') and 'fmin' in self.config and 'fmax' in self.config:
            title += f' (f = {self.config["fmin"]*1e3:.1f} - {self.config["fmax"]*1e3:.1f} mHz)'
        plt.suptitle(title, y=0.95)
        
        plt.tight_layout()
        
        if out:
            return fig