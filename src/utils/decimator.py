#!/bin/python3

"""
Decimation Module

This module provides a class for preparing and processing data for Sagnac demodulation,
including proper decimation with cascading anti-aliasing filters.

Author: Andreas Brotzer
"""

import numpy as np
from scipy import signal
from obspy import Stream, Trace
from typing import Union, Tuple, Optional, List

class Decimator:
    """
    Class for decimation of data.
    
    This class provides methods for decimation, filtering, and general
    signal processing needed before data processing.
    
    Attributes
    ----------
    target_freq : float
        Target sampling frequency in Hz
    detrend : bool
        Whether to apply linear detrending
    taper : bool
        Whether to apply cosine tapering
    taper_fraction : float
        Fraction of data to taper at each end
    filter_before_decim : bool
        Whether to apply bandpass filter before decimation
    filter_after_decim : bool
        Whether to apply bandpass filter after decimation
    filter_freq : tuple of float
        Bandpass filter frequencies (freqmin, freqmax)
    """
    
    def __init__(self,
                 target_freq: float = 1.0,
                 detrend: bool = True,
                 taper: bool = True,
                 taper_fraction: float = 0.05,
                 filter_before_decim: bool = True,
                 filter_after_decim: bool = True,
                 filter_freq: Optional[Tuple[float, float]] = None):
        """
        Initialize the Decimator.
        
        Parameters
        ----------
        target_freq : float, optional
            Target sampling frequency in Hz (default 1.0)
        detrend : bool, optional
            Apply linear detrend before processing (default True)
        taper : bool, optional
            Apply cosine taper before filtering (default True)
        taper_fraction : float, optional
            Taper fraction at each end (default 0.05)
        filter_before_decim : bool, optional
            Apply bandpass filter before decimation (default True)
        filter_after_decim : bool, optional
            Apply bandpass filter after decimation (default True)
        filter_freq : tuple of float, optional
            Bandpass filter frequencies (freqmin, freqmax)
            If None, will be set based on target frequency
        """
        self.target_freq = target_freq
        self.detrend = detrend
        self.taper = taper
        self.taper_fraction = taper_fraction
        self.filter_before_decim = filter_before_decim
        self.filter_after_decim = filter_after_decim
        
        # Set default filter frequencies if not provided
        if filter_freq is None:
            nyquist = target_freq / 2.0
            self.filter_freq = (nyquist * 0.01, nyquist * 0.8)
        else:
            self.filter_freq = filter_freq
    
    @staticmethod
    def _get_prime_factors(n: int) -> List[int]:
        """
        Get prime factors of a number.
        
        Parameters
        ----------
        n : int
            Number to factorize
            
        Returns
        -------
        list of int
            List of prime factors
        """
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d * d > n:
                if n > 1:
                    factors.append(n)
                break
        return factors
    
    def decimate_trace(self, tr: Trace) -> Trace:
        """
        Decimate a trace to target frequency using cascading anti-aliasing filters.
        
        Parameters
        ----------
        tr : obspy.core.trace.Trace
            Input trace to decimate
            
        Returns
        -------
        obspy.core.trace.Trace
            Decimated trace
        """
        # Make a copy to avoid modifying the original
        tr_work = tr.copy()
        
        # Get current sampling rate
        current_freq = tr_work.stats.sampling_rate
        
        # Check if decimation is needed
        if current_freq <= self.target_freq:
            return tr_work
            
        # Calculate decimation factor
        decim_factor = int(current_freq / self.target_freq)
        
        # Get prime factors for cascaded decimation
        factors = self._get_prime_factors(decim_factor)
        
        # Apply cascaded decimation
        for factor in factors:
            # Calculate Nyquist frequency for current stage
            nyquist = current_freq / 2.0
            
            # Design anti-aliasing filter
            # Use 0.8 of Nyquist as cutoff to ensure good anti-aliasing
            freq = 0.8 * (nyquist / factor)
            
            # Apply zero-phase anti-aliasing filter
            tr_work.filter('lowpass', freq=freq, corners=8, zerophase=True)
            
            # Decimate
            tr_work.decimate(factor=factor, no_filter=True)
            
            # Update current frequency
            current_freq = tr_work.stats.sampling_rate
        
        return tr_work
    
    def decimate_stream(self, st: Stream) -> Stream:
        """
        Decimate all traces in a stream to target frequency.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream
            Input stream to decimate
            
        Returns
        -------
        obspy.core.stream.Stream
            Decimated stream
        """
        st_new = Stream()
        for tr in st:
            st_new += self.decimate_trace(tr)
        return st_new
    
    def apply_decimation_stream(self, st: Stream) -> Stream:
        """
        Prepare data for data processing by applying all processing steps.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream
            Input stream to process
            
        Returns
        -------
        obspy.core.stream.Stream
            Processed stream ready for data processing
            
        Notes
        -----
        This method performs the following steps:
        1. Optional detrending
        2. Optional tapering
        3. Optional pre-decimation filtering
        4. Cascaded decimation with anti-aliasing
        5. Optional post-decimation filtering
        """

        # check if target sampling rate is different from current sampling rate
        samplingrates = [tr.stats.sampling_rate for tr in st]
        abort = True
        for samplingrate in samplingrates:
            if samplingrate != self.target_freq:
                abort = False

        # Make a copy to avoid modifying the original
        st_work = st.copy()
        
        if abort:
            print(" -> No resampling required!")
            return st_work

        # Detrend if requested
        if self.detrend:
            st_work.detrend('linear')
            st_work.detrend('demean')
        
        # Taper if requested
        if self.taper:
            st_work.taper(max_percentage=self.taper_fraction, type='cosine')
        
        # Apply pre-decimation filter if requested
        if self.filter_before_decim:
            st_work.filter('bandpass', 
                         freqmin=self.filter_freq[0],
                         freqmax=self.filter_freq[1],
                         corners=4,
                         zerophase=True)
        
        # Perform cascaded decimation
        st_work = self.decimate_stream(st_work)
        
        # Apply post-decimation filter if requested
        if self.filter_after_decim:
            st_work.filter('bandpass',
                         freqmin=self.filter_freq[0],
                         freqmax=self.filter_freq[1],
                         corners=4,
                         zerophase=True)
        
        return st_work
    
    def apply_decimation_trace(self, tr: Trace) -> Trace:
        """
        Prepare data for data processing by applying all processing steps.
        
        Parameters
        ----------
        tr : obspy.core.trace.Trace
            Input stream to process
            
        Returns
        -------
        obspy.core.trace.Trace
            Processed stream ready for data processing
            
        Notes
        -----
        This method performs the following steps:
        1. Optional detrending
        2. Optional tapering
        3. Optional pre-decimation filtering
        4. Cascaded decimation with anti-aliasing
        5. Optional post-decimation filtering
        """
        # Make a copy to avoid modifying the original
        tr_work = tr.copy()
        
        # Detrend if requested
        if self.detrend:
            tr_work.detrend('linear')
            tr_work.detrend('demean')
        
        # Taper if requested
        if self.taper:
            tr_work.taper(max_percentage=self.taper_fraction, type='cosine')
        
        # Apply pre-decimation filter if requested
        if self.filter_before_decim:
            tr_work.filter('bandpass', 
                         freqmin=self.filter_freq[0],
                         freqmax=self.filter_freq[1],
                         corners=4,
                         zerophase=True)
        
        # Perform cascaded decimation
        tr_work = self.decimate_trace(tr_work)
        
        # Apply post-decimation filter if requested
        if self.filter_after_decim:
            tr_work.filter('bandpass',
                         freqmin=self.filter_freq[0],
                         freqmax=self.filter_freq[1],
                         corners=4,
                         zerophase=True)
        
        return tr_work