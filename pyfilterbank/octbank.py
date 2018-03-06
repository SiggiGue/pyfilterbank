# -*- coding: utf-8 -*-
"""This module implements a fractional octave filter bank.
The band passes are realized with butterworth second order sections
described by [Stearns2002]_.
For the second order section filter routines the
module :mod:`sosfiltering` is used.
With the class :class:`FractionalOctaveFilterbank` you can create
filtering objects that apply to the [IEC-61260]_.

An example filter bank is shown by the figures below.

.. plot::

   from pylab import plt
   import octbank
   octbank.example_plot()
   plt.show()


References
----------

.. [Stearns2002] Stearns, Samuel D., Digital Signal Processing with examples in MATLAB
.. [IEC-61260] Electroacoustics - Octave-band and fractional-octave-band filters


Functions
---------
"""
import numpy as np  # TODO: resolve imports for terz fft class...
from numpy import (abs, arange, argmin, array, copy, diff, ones,
                   pi, real, reshape, sqrt, tan, tile, zeros)
from scipy.fftpack import rfft
from pyfilterbank.sosfiltering import (sosfilter_py, sosfilter_double_c,
                       sosfilter_cprototype_py, sosfilter_double_mimo_c)
from pyfilterbank.butterworth import butter_sos

standardized_nominal_frequencies = array([
    0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.6, 3, 0.8,
    1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10,
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
])


def centerfreq_to_bandnum(center_freq, norm_freq, nth_oct):
    """Returns band number from given center frequency."""
    return nth_oct * np.log2(center_freq / norm_freq)


def find_nominal_freq(center_frequencies, nominal_frequencies):
    """Find the nearest nominal frequencies to a given array.

    Parameters
    ----------
    center_frequencies : ndarray
        Some frequencies for those the neares neighbours shall be found.
    nominal_frequencies : ndarray
        The nominal frequencies we want to get the best fitting values to
        `center_frequencies` from.

    Returns
    -------
    nominal_frequencies : generator object
        The neares neighbors nomina freqs to the given frequencies.

    """
    for f in center_frequencies:
        dist = sqrt((standardized_nominal_frequencies - f)**2)
        yield nominal_frequencies[argmin(dist)]


def frequencies_fractional_octaves(start_band, end_band, norm_freq, nth_oct):
    """Return center and band edge frequencies of fractional octaves.

    Parameters
    ----------
    start_band : int
        The starting center frequency at `norm_freq`*2^(`start_band`/`nth_oct`).
    end_band : int
        The last center frequency at `norm_freq`*2^(`end_band`/`nth_oct`).
    norm_freq : scalar
        The center frequency of the band number 0.
    nth_oct : scalar
        The distance between the center frequencies.
        For third octaves `nth_oct=3.0`.

    Returns
    -------
    center_frequencies : ndarray
        Frequencies spaced in `nth_oct` from `start_band` to `end_band`
        with the `norm_freq` at band number 0.
    band_edges : ndarray
        Edge frequencies (-3 dB points) of the fractional octave bands.
        With constant relative Bandwidth.

    """
    k = arange(start_band-1, end_band+2)
    frequencies = norm_freq * 2.0**(k/float(nth_oct))
    band_edges = sqrt(frequencies[:-1] * frequencies[1:])
    center_frequencies = frequencies[1:-1]
    return center_frequencies, band_edges


def to_normalized_frequencies(frequencies, sample_rate, clip=True):
    """Returns normalized frequency array.

    Parameters
    ----------
    frequencies : ndarray
        Vector with given frequencies.
    sample_rate : scalar
        The sample rate. Frequencies beyond Nyquist criterion
        will be truncated.

    Returns
    -------
    normalized_frequencies : ndarray
        Normalized, truncated frequency array.
    """
    index_nyquis = frequencies >= 0.5*sample_rate
    freqs = copy(frequencies)
    if clip and any(index_nyquis):
        freqs[index_nyquis] = 0.499*sample_rate
        return freqs[:list(index_nyquis).index(True)+1] / sample_rate
    else:
        return freqs[~index_nyquis] / sample_rate


def design_sosmat_band_passes(order, band_edges, sample_rate,
                              edge_correction_percent=0.0):
    """Return matrix containig sos coeffs of bandpasses.

    Parameters
    ----------
    order : int
        Order of the band pass filters.
    band_edges : ndarray
        Band edge frequencies for the bandpasses.
    sample_rate : scalar
        Sample frequency.
    edge_correction_percent : scalar
        Percentage for the correction of the bandedges.
        Float between -100 % and 100 %.
        It can be helpfull dependent on the used filter order.
        p > 0 widens the band passes.

    Returns
    -------
    sosmat : ndarray
        Second order section coefficients.
        Each column is one band pass cascade of coefficients.
    """
    num_coeffs_biquad_bandpass = 6
    num_coeffs_cascade = order * num_coeffs_biquad_bandpass
    num_bands = len(band_edges) - 1
    sosmat = zeros((num_coeffs_cascade, num_bands))

    band_edges_normalized = to_normalized_frequencies(band_edges, sample_rate)
    p_lower = (1 - edge_correction_percent*1e-2)
    p_upper = (1 + edge_correction_percent*1e-2)

    for i, (lower_freq, upper_freq) in enumerate(zip(
            band_edges_normalized[:-1],
            band_edges_normalized[1:])):
        sos = butter_sos('bandpass',
                         order,
                         p_lower*lower_freq,
                         p_upper*upper_freq)
        sosmat[:, i] = sos.flatten()
    return sosmat


def design_sosmat_low_pass_high_pass_bounds(order, band_edges, sample_rate):
    """Returns matrix containing sos coeffs of low and highpass.
    The cutoff frequencies are placed at the first and last band edge.

    .. note:: This funtion is not used anymore.

    Parameters
    ----------
    order : int
        Order of the band pass filters.
    band_edges : ndarray
        Band edge frequencies for the low an highpass.
    sample_rate : scalar
        Sample rate.

    Returns
    -------
    sosdict : ndarray
        Second order section coefficients,
        the first column contains the low pass coefs
        and the second column contains the highpass coeffs.

    """
    sosmat = zeros((0.5*order*6, 2))
    band_edges_normalized = to_normalized_frequencies(band_edges, sample_rate)

    sosmat[:, 0] = butter_sos('lowpass', order,
                              band_edges_normalized[0]).flatten()

    sosmat[:, 1] = butter_sos('highpass', order,
                              band_edges_normalized[-1]).flatten()
    return sosmat


class FractionalOctaveFilterbank:
    """Fractional octave filter bank
    with second order section butterworth band passes.

    Parameters
    ----------
    sample_rate : int
        Sampling rate of the signals to be filtered.
    order : int
        Filter order of the bands. As this are second order sections, it
        has to be even. Otherweise you'll get an error.
    nth_oct : scalar
        Number of bands per octave.
    norm_freq : scalar
        This is the reference frequency for all fractional octaves
        placed around this band.
    start_band : int
        First Band number of fractional octaves below `norm_freq`.
    end_band : int
        Last band number of fractional octaves above `norm_freq`.
    edge_correction_percent : scalar
        Percentage of widening or narrowing the bands.
    filterfun : {'cffi', 'py', 'cprototype'}
        Function used by the method :func:`filter`.

    Attributes
    ----------
    center_frequencies : ndarray
    band_edges : ndarray
        Frequencies at -3 dB point for all band passes.
        This are the cross sections of the bands if no edge correction
        applied.
    sosmat : ndarray
        Filter coefficient matrix with second order section band passes.
    num_bands : int
        Number of frequency bands in the filter bank.
    band_widths : ndarray
        The -3 dB band width of each band pass in the filter bank.
    effective_filter_lengths : ndarray
        The effective length of the filters in seconds.
        A filtered block should at least have same length
        if you want to avoid energy leakage.

    Examples
    --------
    >>> from pyfilterbank import FractionalOctaveFilterbank
    >>> from pylab import plt, np
    >>>
    >>> sample_rate = 44100
    >>> ofb = FractionalOctaveFilterbank(sample_rate, order=4)
    >>>
    >>> x = np.random.randn(4*sample_rate)
    >>> y, states = ofb.filter(x)
    >>> L = 10 * np.log10(np.sum(y*y,axis=0))
    >>> plt.plot(L)

    """
    def __init__(self,
                 sample_rate=44100,
                 order=4,
                 nth_oct=3.0,
                 norm_freq=1000.0,
                 start_band=-19,
                 end_band=13,
                 edge_correction_percent=0.01,
                 filterfun='cffi'):
        self._sample_rate = sample_rate
        self._order = order
        self._nth_oct = nth_oct
        self._norm_freq = norm_freq
        self._start_band = start_band
        self._end_band = end_band
        self._edge_correction_percent = edge_correction_percent
        self._initialize_filter_bank()
        self.set_filterfun(filterfun)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value
        self._initialize_filter_bank()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._initialize_filter_bank()

    @property
    def nth_oct(self):
        return self._nth_oct

    @nth_oct.setter
    def nth_oct(self, value):
        self._nth_oct = value
        self._initialize_filter_bank()

    @property
    def norm_freq(self):
        return self._norm_freq

    @norm_freq.setter
    def norm_freq(self, value):
        self._norm_freq = value
        self._initialize_filter_bank()

    @property
    def start_band(self):
        return self._start_band

    @start_band.setter
    def start_band(self, value):
        self._start_band = value
        self._initialize_filter_bank()

    @property
    def end_band(self):
        return self._end_band

    @end_band.setter
    def end_band(self, value):
        self._end_band = value
        self._initialize_filter_bank()

    @property
    def edge_correction_percent(self):
        return self._edge_correction_percent

    @edge_correction_percent.setter
    def edge_correction_percent(self, value):
        self._edge_correction_percent = value
        self._initialize_filter_bank()

    @property
    def center_frequencies(self):
        return self._center_frequencies

    @property
    def band_edges(self):
        return self._band_edges

    @property
    def sosmat(self):
        return self._sosmat

    @property
    def num_bands(self):
        return len(self.center_frequencies)

    @property
    def band_widths(self):
        return diff(self.band_edges)

    @property
    def effective_filter_lengths(self):
        """Returns an estimate of the effective filter length"""
        return [int(l) for l in self.sample_rate*3//self.band_widths]

    def _initialize_filter_bank(self):
        center_frequencies, band_edges = frequencies_fractional_octaves(
            self.start_band, self.end_band,
            self.norm_freq, self.nth_oct
        )
        self._center_frequencies = center_frequencies
        self._band_edges = band_edges

        sosmat_band_passes = design_sosmat_band_passes(
            self.order, self.band_edges,
            self.sample_rate, self.edge_correction_percent
        )
        self._sosmat = sosmat_band_passes

    def set_filterfun(self, filterfun_name):
        """Set the function that is used for filtering
        with the method `self.filter`.

        Parameters
        ----------
        filterfun_name : {'cffi', 'py', 'cprototype'}
            Three different filter functions,
            'cffi' is the fastest, 'py' is implemented with `lfilter`.

        """
        filterfun_name = filterfun_name.lower()
        if filterfun_name == 'cffi':
            self.sosfilterfun = sosfilter_double_c
            self.filterfun_name = filterfun_name
        elif filterfun_name == 'py':
            self.sosfilterfun = sosfilter_py
            self.filterfun_name = filterfun_name
        elif filterfun_name == 'cprototype':
            self.sosfilterfun = sosfilter_cprototype_py
            self.filterfun_name = filterfun_name
        else:
            print('Could not change filter function.')

    def filter_mimo_c(self, x, states=None):
        """Filters the input by the settings of the filterbank object.

        It supports multi channel audio and returns a 3-dim ndarray.
        Only for real valued signals.
        No ffilt (backward forward filtering) implemented in this method.

        Parameters
        ----------
        x : ndarray
            Signal to be filtered.
        states : ndarray or None
            States of the filter sections (for block processing).

        Returns
        --------
        signal : ndarray
            Signal array (NxBxC), with N samples, B frequency bands
            and C-signal channels.
        states : ndarray
            Filter states of all filter sections.
        """
        return sosfilter_double_mimo_c(x, self.sosmat, states)

    def filter(self, x, ffilt=False, states=None):
        """Filters the input by the settings of the filterbank object.

        Parameters
        ----------
        x :  ndarray
            Input signal (Nx0)
        ffilt : bool
            Forward and backward filtering, if Ture.
        states : dict
            States of all filter sections in the filterbank.
            Initial you can states=None before block process.

        Returns
        -------
        y : ndarray
            Fractional octave signals of the filtered input x
        states : dict
            Dictionary containing all filter section states.
        """
        # in the next version this will be turne to a multi dimensional np array
        y_data = zeros((len(x), len(self.center_frequencies)))

        if not isinstance(states, dict):
            states_allbands = dict()
            for f in self.center_frequencies: states_allbands[f] = None
        else :
            states_allbands = states

        for i, f in enumerate(self.center_frequencies):
            states = states_allbands[f]
            sos = reshape(self.sosmat[:, i], (self.order, 6))
            if not ffilt:
                y, states = self.sosfilterfun(x.copy(), sos, states)
            elif ffilt:
                y, states = self.sosfilterfun(x.copy()[::-1], sos, states)
                y, states = self.sosfilterfun(y[::-1], sos, states)

            y_data[:, i] = y
            states_allbands[f] = states
        return y_data, states_allbands


def freqz(ofb, length_sec=6, ffilt=False, plot=True):
    """Computes the IR and FRF of a digital filter.

    Parameters
    ----------
    ofb : FractionalOctaveFilterbank object
    length_sec : scalar
        Length of the impulse response test signal.
    ffilt : bool
        Backard forward filtering. Effectiv order is doubled then.
    plot : bool
        Create Plots or not.

    Returns
    -------
    x : ndarray
        Impulse test signal.
    y : ndarray
        Impules responses signal of the filters.
    f : ndarray
        Frequency vector for the FRF.
    Y : Frequency response (FRF) of the summed filters.

    """
    from pylab import np, plt, fft, fftfreq
    x = np.zeros(length_sec*ofb.sample_rate)
    x[int(length_sec*ofb.sample_rate/2)] = 0.9999

    if not ffilt:
        y, states = ofb.filter_mimo_c(x)
        y = y[:, :, 0]
    else:
        y, states = ofb.filter(x, ffilt=ffilt)
    s = np.zeros(len(x))
    len_x_2 = int(len(x)/2)
    for i in range(y.shape[1]):
        s += y[:, i]
        X = fft(y[:, i])  # sampled frequency response
        f = fftfreq(len(x), 1.0/ofb.sample_rate)
        if plot:
            fig = plt.figure('freqz filter bank')
            plt.grid(True)
            plt.axis([0, ofb.sample_rate / 2, -100, 5])

            L = 20*np.log10(np.abs(X[:len_x_2]) + 1e-17)
            plt.semilogx(f[:len_x_2], L, lw=0.5)

    Y = fft(s)
    if plot:
        plt.title(u'freqz() Filter Bank')
        plt.xlabel('Frequency / Hz')
        plt.ylabel(u'Damping /dB(FS)')
        plt.xlim((10, ofb.sample_rate/2))
        plt.figure('sum')
        L = 20*np.log10(np.abs(Y[:len_x_2]) + 1e-17)
        plt.semilogx(f[:len_x_2], L, lw=0.5)

        level_input = 10*np.log10(np.sum(x**2))
        level_output = 10*np.log10(np.sum(s**2))
        plt.axis([5, ofb.sample_rate/1.8, -50, 5])
        plt.grid(True)
        plt.title('Sum of filter bands')
        plt.xlabel('Frequency / Hz')
        plt.ylabel(u'Damping /dB(FS)')

        print('sum level', level_output, level_input)

    return x, y, f, Y


class ThirdOctFFTLevel:

    """Third octave levels by fft.
    TODO: rename variables
    TODO: Write Documentation
    """

    def __init__(self,
                 fmin=30,
                 fmax=17000,
                 nfft=16384,
                 fs=44100,
                 flag_mean=False):
        self.nfft = nfft
        self.fs = fs

        # following should go into some functions:
        kmin = 11 + int(10*np.log10(fmin))
        kmax = 11 + int(10*np.log10(fmax))
        f_terz = standardized_nominal_frequencies[kmin:kmax]
        n = int(1 + kmax - kmin)
        halfbw = 2**(1.0/6)
        df = fs/nfft
        idx_lower = np.zeros(n)
        idx_lower[0] = 10 + np.round((
            standardized_nominal_frequencies[kmin]/halfbw)/df)

        idx_upper = 10 + np.round(
            halfbw*standardized_nominal_frequencies[kmin:kmax]/df)
        idx_lower[1:n] = idx_upper[0:n-1] + 1

        upperedge = halfbw * standardized_nominal_frequencies[kmax]
        print(idx_upper[0]-idx_lower[0])
        #if idx_upper(1) - idx_lower(1) < 4:
        #    raise ValueError('Too few FFT lines per frequency band')

        M = np.zeros((n, int(nfft/2)+1))

        for cc in range(n-1):
            kk = range(int(idx_lower[cc]), int(idx_upper[cc]))
            M[cc, kk] = 2.0/(self.nfft/2+1)
            if kk[0] == 0:
                M[cc, kk[0]] = 1.0/(self.nfft/2+1)

        self.M = M
        self.f_terz = f_terz

    def filter(self, x):
        Xsq = np.abs(rfft(x, self.nfft/2 + 1))**2
        return 10*np.log10(np.dot(self.M, Xsq))


def print_parseval(x, X):
    print(np.sum(x*x))
    print(np.sum(X*X))


def example_plot():
    """Creates a plot with :func:`freqz` of the default
    :class:`FractionalOctaveFilterbank`.
    """
    ofb = FractionalOctaveFilterbank()
    x, y, f, Y = freqz(ofb)
