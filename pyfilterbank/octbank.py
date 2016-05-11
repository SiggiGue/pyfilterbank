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

.. [Stearns2002] Stearns, Samuel D., Digital Signal Processing
                 with examples in MATLAB
.. [IEC-61260] Electroacoustics- Octave-band and fractional-octave-band filters


Functions
---------
"""

from itertools import cycle

import numpy as np
from scipy.fftpack import rfft
from scipy.signal import decimate

from pyfilterbank.sosfiltering import (sosfilter_py,
                                       sosfilter_double_c,
                                       sosfilter_cprototype_py,
                                       sosfilter_double_mimo_c)
from pyfilterbank.butterworth import butter_sos


standardized_nominalfreqs = np.array([
    0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.6, 3, 0.8,
    1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10,
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
])


def centerfreq_to_bandnum(centerfreq, normfreq, fraction):
    """Returns band number from given center frequency."""
    return fraction * np.log2(centerfreq / normfreq)


def find_nominal_freq(centerfreqs, nominalfreqs):
    """Find the nearest nominal frequencies to a given array.

    Parameters
    ----------
    centerfreqs : ndarray
        Some frequencies for those the neares neighbours shall be found.
    nominalfreqs : ndarray
        The nominal frequencies we want to get the best fitting values to
        `centerfreqs` from.

    Returns
    -------
    nominalfreqs : generator object
        The neares neighbors nomina freqs to the given frequencies.

    """
    for f in centerfreqs:
        dist = np.sqrt((standardized_nominalfreqs - f)**2)
        yield nominalfreqs[np.argmin(dist)]


def frequencies_fractional_octaves(
        startband,
        endband,
        normfreq,
        fraction,
        base=10):
    """Returns center and band edge frequencies of fractional octaves.

    Parameters
    ----------
    startband : int
        The starting center frequency at `normfreq`*2^(`startband`/`fraction`)
    endband : int
        The last center frequency at `normfreq`*2^(`endband`/`fraction`).
    normfreq : scalar
        The center frequency of the band number 0.
    fraction : scalar
        The number of bands per octave.
        For third octaves `fraction=3` and for octaves `fraction=1`.
    base : int {10, 2}
        Base number for calkulation of center frequencies. Default is 10.

    Returns
    -------
    centerfreqs : ndarray
        Frequencies spaced in `fraction` from `startband` to `endband`
        with the `normfreq` at band number 0.
    bandedges : ndarray
        Edge frequencies (-3 dB points) of the fractional octave bands.
        With constant relative Bandwidth.

    """
    k = np.arange(startband-1, endband+2)
    if int(base) == 2:
        frequencies = normfreq * 2.0**(k/fraction)
    elif int(base) == 10:
        frequencies = normfreq * 10.0**(0.3 * k/fraction)
    else:
        raise ValueError('`base` must be 10 or 2, not {}.'.format(base))

    bandedges = np.sqrt(frequencies[:-1] * frequencies[1:])
    centerfreqs = frequencies[1:-1]
    return centerfreqs, bandedges


def to_normalized_frequencies(frequencies, samplerate, clip=True):
    """Returns a normalized frequency array.

    Parameters
    ----------
    frequencies : ndarray
        Vector with given frequencies.
    samplerate : scalar
        The sample rate. Frequencies beyond Nyquist criterion
        will be truncated.

    Returns
    -------
    normalized_frequencies : ndarray
        Normalized, truncated frequency array.
    """
    index_nyquis = frequencies >= 0.5 * samplerate
    freqs = np.copy(frequencies)
    if clip and any(index_nyquis):
        freqs[index_nyquis] = 0.499 * samplerate
        return freqs[:list(index_nyquis).index(True)+1] / samplerate
    else:
        return freqs[~index_nyquis] / samplerate


def design_sosmat_band_passes(
        order,
        bandedges,
        samplerate,
        edge_correction_percent=0.0):
    """Return matrix containig sos coeffs of bandpasses.

    Parameters
    ----------
    order : int
        Order of the band pass filters.
    bandedges : ndarray
        Band edge frequencies for the bandpasses.
    samplerate : scalar
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
    numbands = len(bandedges) - 1
    sosmat = np.zeros((num_coeffs_cascade, numbands))

    bandedges_normalized = to_normalized_frequencies(bandedges, samplerate)
    p_lower = (1 - edge_correction_percent*1e-2)
    p_upper = (1 + edge_correction_percent*1e-2)

    for i, (lower_freq, upper_freq) in enumerate(zip(
            bandedges_normalized[:-1],
            bandedges_normalized[1:])):

        sos = butter_sos(
            'bandpass',
            order,
            p_lower*lower_freq,
            p_upper*upper_freq)
        sosmat[:, i] = sos.flatten()

    return sosmat


def design_sosmat_low_pass_high_pass_bounds(order, bandedges, samplerate):
    """Returns matrix containing sos coeffs of low and highpass.
    The cutoff frequencies are placed at the first and last band edge.

    .. note:: This funtion is not used anymore.

    Parameters
    ----------
    order : int
        Order of the band pass filters.
    bandedges : ndarray
        Band edge frequencies for the low an highpass.
    samplerate : scalar
        Sample rate.

    Returns
    -------
    sosdict : ndarray
        Second order section coefficients,
        the first column contains the low pass coefs
        and the second column contains the highpass coeffs.

    """
    sosmat = np.zeros((0.5*order*6, 2))
    bandedges_normalized = to_normalized_frequencies(bandedges, samplerate)

    sosmat[:, 0] = butter_sos('lowpass',
                              order,
                              bandedges_normalized[0]).flatten()

    sosmat[:, 1] = butter_sos('highpass',
                              order,
                              bandedges_normalized[-1]).flatten()
    return sosmat


class FractionalOctaveFilterbank:
    """Fractional octave filter bank
    with second order section butterworth band passes.

    Parameters
    ----------
    samplerate : int
        Sampling rate of the signals to be filtered.
    order : int
        Filter order of the bands. As this are second order sections, it
        has to be even. Otherweise you'll get an error.
    fraction : scalar
        Number of bands per octave.
    normfreq : scalar
        This is the reference frequency for all fractional octaves
        placed around this band.
    startband : int
        First Band number of fractional octaves below `normfreq`.
    endband : int
        Last band number of fractional octaves above `normfreq`.
    edge_correction_percent : scalar
        Percentage of widening or narrowing the bands.
    filterfun : {'cffi', 'py', 'cprototype'}
        Function used by the method :func:`filter`.

    Attributes
    ----------
    centerfreqs : ndarray
    bandedges : ndarray
        Frequencies at -3 dB point for all band passes.
        This are the cross sections of the bands if no edge correction
        applied.
    sosmat : ndarray
        Filter coefficient matrix with second order section band passes.
    numbands : int
        Number of frequency bands in the filter bank.
    bandwidths : ndarray
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
    >>> samplerate = 44100
    >>> ofb = FractionalOctaveFilterbank(samplerate, order=4)
    >>>
    >>> x = np.random.randn(4*samplerate)
    >>> y, states = ofb.filter(x)
    >>> L = 10 * np.log10(np.sum(y*y,axis=0))
    >>> plt.plot(L)

    """
    def __init__(self,
                 samplerate=44100,
                 order=4,
                 fraction=3.0,
                 normfreq=1000.0,
                 startband=-19,
                 endband=13,
                 edge_correction_percent=0.01,
                 filterfun='cffi'):
        self._samplerate = samplerate
        self._order = order
        self._fraction = fraction
        self._normfreq = normfreq
        self._startband = startband
        self._endband = endband
        self._edge_correction_percent = edge_correction_percent
        self._initialize_filter_bank()
        self.set_filterfun(filterfun)

    @property
    def samplerate(self):
        return self._samplerate

    @samplerate.setter
    def samplerate(self, value):
        self._samplerate = value
        self._initialize_filter_bank()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._initialize_filter_bank()

    @property
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        self._fraction = value
        self._initialize_filter_bank()

    @property
    def normfreq(self):
        return self._normfreq

    @normfreq.setter
    def normfreq(self, value):
        self._normfreq = value
        self._initialize_filter_bank()

    @property
    def startband(self):
        return self._startband

    @startband.setter
    def startband(self, value):
        self._startband = value
        self._initialize_filter_bank()

    @property
    def endband(self):
        return self._endband

    @endband.setter
    def endband(self, value):
        self._endband = value
        self._initialize_filter_bank()

    @property
    def edge_correction_percent(self):
        return self._edge_correction_percent

    @edge_correction_percent.setter
    def edge_correction_percent(self, value):
        self._edge_correction_percent = value
        self._initialize_filter_bank()

    @property
    def centerfreqs(self):
        return self._centerfreqs

    @property
    def bandedges(self):
        return self._bandedges

    @property
    def sosmat(self):
        return self._sosmat

    @property
    def numbands(self):
        return len(self.centerfreqs)

    @property
    def bandwidths(self):
        return np.diff(self.bandedges)

    @property
    def effective_filter_lengths(self):
        """Returns an estimate of the effective filter length"""
        return [int(l) for l in self.samplerate*3//self.bandwidths]

    def _initialize_filter_bank(self):
        centerfreqs, bandedges = frequencies_fractional_octaves(
            self.startband, self.endband,
            self.normfreq, self.fraction
        )
        self._centerfreqs = centerfreqs
        self._bandedges = bandedges

        sosmat_band_passes = design_sosmat_band_passes(
            self.order, self.bandedges,
            self.samplerate, self.edge_correction_percent
        )
        self._sosmat = sosmat_band_passes

    def set_filterfun(self, filterfun_name):
        """Set the function that is used for filtering
        with the method `self.filter`.

        Parameters
        ----------
        filterfun_name : {'cffi', 'scipy', 'cprototype'}
            Three different filter functions,
            'cffi' is the fastest, 'py' is implemented with `lfilter`.

        """
        filterfun_name = filterfun_name.lower()
        if filterfun_name == 'cffi':
            self.sosfilterfun = sosfilter_double_c
            self.filterfun_name = filterfun_name
        elif filterfun_name == 'scipy':
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

        Uses the self.filterfun function to filter the signal.

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

        y_data = np.zeros((len(x), len(self.centerfreqs)))

        if not isinstance(states, dict):
            states_allbands = dict()
            for f in self.centerfreqs:
                states_allbands[f] = None
        else:
            states_allbands = states

        for i, f in enumerate(self.centerfreqs):
            states = states_allbands[f]
            sos = np.reshape(self.sosmat[:, i], (self.order, 6))

            if not ffilt:
                y, states = self.sosfilterfun(x.copy(), sos, states)
            elif ffilt:
                y, states = self.sosfilterfun(x.copy()[::-1], sos, states)
                y, states = self.sosfilterfun(y[::-1], sos, states)

            y_data[:, i] = y
            states_allbands[f] = states
        return y_data, states_allbands

    def mrfilter(self, x,  states=None, conserveenergy=True):
        """Filters the input by multi rate.

        The upper 1.33 Octaves will be filtered at self.samplerate but
        the remaining bands will be decimated by 2 per octave. The
        33%% plus to the octave is filtered in original sample rate because
        of the band distortion near to the nyquist frequency. This shall be
        a problem only in the very highest band but not in all last bands
        of the octaves below.

        Parameters
        ----------
        x :  ndarray
            Input signal (Nx0)
        states : dict
            States of all filter sections in the filterbank.
            Initial you can states=None before block process.
        conserveenergy : bool
            Adjusts signal length if filter lenght is greater by appending 0.

        Returns
        -------
        y : list of multirate signals
            Fractional octave signals of the filtered input x
        states : dict
            Dictionary containing all filter section states.
        nframesl : list of int
            Actual number of samples in due to downsampling.
            Is needed if you want to calculate the real energy due to appended
            zeros if conserveenergy=True.

        """

        if not isinstance(states, dict):
            states_allbands = dict()
            for f in self.centerfreqs:
                states_allbands[f] = None
        else:
            states_allbands = states

        # Original rate section
        upper = int(self.fraction + np.ceil(self.fraction*0.33))
        uppercfreqs = reversed(self.centerfreqs[-upper:])
        uppersosl = reversed(list(self.sosmat[:, -upper:].T))

        # Multi rate section
        mrcfreqs = reversed(self.centerfreqs[:-upper])
        mrcsosmat = list(self.sosmat[:, -upper:-upper+int(self.fraction)].T)
        mrcsosl = reversed([np.reshape(s, (self.order, 6)) for s in mrcsosmat])
        mrfl = reversed(self.effective_filter_lengths[:-upper])
        ydata = []
        nframesl = []

        # Filter the upper 1.33 octaves with original sample rate:
        for sos, f in zip(uppersosl, uppercfreqs):
            states = states_allbands[f]
            sos = np.reshape(sos, (self.order, 6))
            y, states = self.sosfilterfun(x.copy(), sos, states)
            ydata += [y]
            states_allbands[f] = states
            nframesl += [len(x)]

        # Filter the remaining bans with multirate approach:
        nd = 0
        for c, (sos, f, fl) in enumerate(zip(cycle(mrcsosl), mrcfreqs, mrfl)):
            if c == 0 or (c % self.fraction) == 0:
                x = decimate(x, 2)
                nd += 1
                nframes = len(x)
                if conserveenergy and nframes < fl/nd:
                    # Append zeros so the signal is as long as the effective
                    # filter response length:
                    xf = np.concatenate((x, np.zeros(int(fl/nd)-nframes)))
                else:
                    xf = x
            states = states_allbands[f]
            y, states = self.sosfilterfun(xf, sos, states)
            ydata += [y]
            states_allbands[f] = states
            nframesl += [nframes]
        return list(reversed(ydata)), states_allbands, list(reversed(nframesl))

    def process_levels(self, x, conserveenergy=True):
        """Returns the level of each band by using mrfilter.

        Parameters
        ----------
        x : ndarray
        conserveenergy : bool
            Appends zeros if signal is to short for effective filter length.

        Returns
        -------
        levels : narray

        """
        def level(b, nfr):
            return 10 * np.log10(np.sum(b*b)/nfr)
        y, states, nframesl = self.mrfilter(x, conserveenergy=conserveenergy)
        return np.array([level(b, nf) for b, nf in zip(y, nframesl)])


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
    x = np.zeros(length_sec*ofb.samplerate)
    x[length_sec*ofb.samplerate/2] = 0.9999
    if not ffilt:
        y, states = ofb.filter_mimo_c(x)
        y = y[:, :, 0]
    else:
        y, states = ofb.filter(x, ffilt=ffilt)
    s = np.zeros(len(x))
    for i in range(y.shape[1]):
        s += y[:, i]
        X = fft(y[:, i])  # sampled frequency response
        f = fftfreq(len(x), 1.0/ofb.samplerate)
        if plot:
            plt.figure('freqz filter bank')
            plt.grid(True)
            plt.axis([0, ofb.samplerate / 2, -100, 5])
            L = 20*np.log10(np.abs(X[:len(x)/2]) + 1e-17)
            plt.semilogx(f[:len(x)/2], L, lw=0.5)
            plt.hold(True)

    Y = fft(s)
    if plot:
        plt.title(u'freqz() Filter Bank')
        plt.xlabel('Frequency / Hz')
        plt.ylabel(u'Damping /dB(FS)')
        plt.xlim((10, ofb.samplerate/2))
        plt.hold(False)

        plt.figure('sum')
        L = 20*np.log10(np.abs(Y[:len(x)/2]) + 1e-17)
        plt.semilogx(f[:len(x)/2], L, lw=0.5)
        level_input = 10*np.log10(np.sum(x**2))
        level_output = 10*np.log10(np.sum(s**2))
        plt.axis([5, ofb.samplerate/1.8, -50, 5])
        plt.grid(True)
        plt.title('Sum of filter bands')
        plt.xlabel('Frequency / Hz')
        plt.ylabel(u'Damping /dB(FS)')

        print('sum level', level_output, level_input)

    return x, y, f, Y


class ThirdOctFFTLevel:

    """Third octave levels by fft.
    EXPERIMENTAL not TESTED.
    TODO: rename variables
    TODO: Write Documentation
    """

    def __init__(self,
                 fmin=30,
                 fmax=20000,
                 nfft=4*16384,
                 fs=44100,
                 flag_mean=False):
        self.nfft = nfft
        self.fs = fs

        # following should go into some functions:
        kmin = int(11 + 10 * np.log10(fmin))
        kmax = int(11 + 10 * np.log10(fmax))
        centerfrqs = standardized_nominalfreqs[kmin:kmax+1]
        n = int(kmax - kmin)
        # halfbw = 2**(1.0/6)
        halfbw = 10**(0.3/6)
        df = fs / nfft
        idx_lower = np.zeros(n)
        idx_lower[0] = np.round((
            standardized_nominalfreqs[kmin] / halfbw) / df)

        idx_upper = np.round(
            halfbw * standardized_nominalfreqs[kmin:kmax] / df)
        idx_lower[1:] = idx_upper[:n-1] + 1

        # upperedge = halfbw * standardized_nominalfreqs[kmax]
        if (idx_upper[0] - idx_lower[0]) < 4:
            print(idx_upper[0] - idx_lower[0])
            raise ValueError('Too few FFT lines per frequency band')

        if self.nfft % 2 == 0:
            M = np.zeros((n, nfft/2+1))
        else:
            M = np.zeros((n, (nfft+1)/2))

        for cc in range(n):
            kk = range(int(idx_lower[cc]), int(idx_upper[cc]))
            if not flag_mean:
                M[cc, kk] = 1.0
            else:
                M[cc, kk] = 1.0 / len(kk)

        self.M = M
        self.centerfrqs = centerfrqs

    def filter(self, x):
        X = rfft(x, self.M.shape[1])
        return (10*np.log10(np.dot(self.M, X*X)) +
                10*np.log10(2/self.fs/self.M.shape[1]))


def example_plot():
    """Creates a plot with :func:`freqz` of the default
    :class:`FractionalOctaveFilterbank`.
    """
    ofb = FractionalOctaveFilterbank()
    x, y, f, Y = freqz(ofb)
