# -*- coding: utf-8 -*-
"""This module implements fractional octaves and filters.

The band passes are realized with butterworth second order sections
described by [Stearns2002]_ per default.

For the second order section filter routines the
module :mod:`sosfiltering` is used.

With the class :class:`*FractionalOctaveFilter*` you can create
filtering objects that apply to the [IEC-61260]_.

An example filter bank is shown by the figures below.

.. plot::

   from pylab import plt
   import octave
   octave.example_plot()
   plt.show()


References
----------

.. [Stearns2002] Stearns, Samuel D., Digital Signal Processing
                 with examples in MATLAB
.. [IEC-61260] Electroacoustics - Octave-band and fractional-octave-band filters


"""
import warnings
from itertools import cycle
from math import log2, log10
import numpy as np
from scipy.fftpack import rfft
from scipy.signal import decimate

from pyfilterbank import sosfiltering
from pyfilterbank.sosfiltering import (sosfilter_py,
                                       sosfilter_double_c,
                                       sosfilter_cprototype_py,
                                       sosfilter_double_mimo_c)

from . import butterworth
from . import chebyshev


STANDARDIZED_NOMINALFREQS = np.array([
    100, 112, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000
])
DEFAULT_REFERENCEFREQ = 1000.0
DEFAULT_FRACTION = 1.0
DEFAULT_OCTAVE_BASE = 10.0
__NUM_COEFFS_BIQUAD_BANDPASS = 6


class BaseFractionalOctaveModel:

    def __init__(self,  
                 centerfreq=None, 
                 fraction=DEFAULT_FRACTION, 
                 base=DEFAULT_OCTAVE_BASE,
                 **kwargs):
        """Base Model of Fractional Octave Band Frequencies

        Properties
        ----------
        centerfreq : scalar
            Center frequency of the band. 
        fraction : scalar
            One octave fraction is 1, third octave fraction is 3 etc.
            Sets 1/fraction octave band width.
            Default is 1.0
        base : int {10, 2}
            Default base for frequency calculations.
            Be aware that only base=10 complies IEC 61260:2014.

        """
        self._centerfreq = centerfreq
        self._fraction = fraction
        self._base = base    
        self._loweredgefreq = None
        self._upperedgefreq = None
        self._halfbw_factor = None
        self._loweredgefreq = None
        self._upperedgefreq = None
        self._bandwidth = None
        self._relative_bandwidth = None
        self._update()

    def _update(self):
        """Updates the state of self, so all properties become refreshed."""
        # print('Update', 'BaseFractionalOctaveModel')
        if self._centerfreq is None:
            # it seems the user did not set the centerfreq to a valid value.
            return
        centerfreq, halfbw_factor = self._centerfreq, self._get_halfbw_factor()
        self._halfbw_factor = halfbw_factor
        self._loweredgefreq = centerfreq / halfbw_factor 
        self._upperedgefreq = centerfreq * halfbw_factor
        self._bandwidth = self._upperedgefreq - self._loweredgefreq
        self._relative_bandwidth = self._bandwidth / self._centerfreq

    @property
    def base(self):
        """Returns the octave base value {2, 10}"""
        return self._base

    @base.setter
    def base(self, value):
        """Changes the octave base value, allowed is 2 or 10.
        
        Only 10 complies to IEC 61260.

        """
        value = int(value)
        if value == 2  or value == 10:
            self._base = value
            self._update()
        else:
            raise ValueError('`base` must be 10 or 2, not {}.'.format(value))        

    @property
    def centerfreq(self):
        """Returns the center frequency."""
        return self._centerfreq
    
    @centerfreq.setter
    def centerfreq(self, value):
        """Changes the center frequency. It must be scalar and greater than 0."""
        if np.isscalar(value):
            if value <= 0:
                raise ValueError('Given center frequency must be greater than 0.')
            self._centerfreq = float(value)
            self._update()
        else:
            raise ValueError('`centerfreq` must be scalar not {}'.format(value))

    @property
    def fraction(self):
        """Returns the octave fraction."""
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        """Changes the octave fraction."""
        if np.isscalar(value):
            self._fraction = float(value)
            self._update()
        else:
            raise ValueError('`fraction` must be scalar not {}'.format(value))


    def _get_halfbw_factor(self):
        """Returns the half bandwidth factor."""
        if self.fraction is not None:
            if self.base == 10:
                return 10**(0.3/(2*self.fraction))
            else:
                return 2**(1.0/(2*self.fraction))
        else:
            raise ValueError('Please assign a value to `self.fraction`.')

    @property
    def loweredgefreq(self):
        """Returns the lower edge frequency.
        
        I.e. the -3 dB edge frequency below centerfreq.
        
        """
        return self._loweredgefreq

    @property
    def upperedgefreq(self):
        """Returns the upper edge frequency.
        
        I.e. the -3 dB edge frequency above centerfreq.
        
        """
        return self._upperedgefreq

    @property
    def bandwidth(self):
        """Returns the bandwidth."""
        return self._bandwidth

    @property
    def relative_bandwidth(self):
        """Returns the bandwidth related to the center frequency."""
        return self._relative_bandwidth

    def __repr__(self):
        return '<{}: 1/{} Oct.@{:.1f} Hz ({:.1f} to {:.1f} Hz)'.format(
            self.__class__.__name__, 
            self.fraction, 
            self.centerfreq, 
            self.loweredgefreq,
            self.upperedgefreq
            )


class FractionalOctaveModel(BaseFractionalOctaveModel):

    def __init__(self, 
                 bandnum=None, 
                 fraction=DEFAULT_FRACTION, 
                 referencefreq=DEFAULT_REFERENCEFREQ, 
                 base=DEFAULT_OCTAVE_BASE,
                 **kwargs):
        """Fractional Octave Band Frequencies according to IEC 61260:2014.

        If you change a parameter except centerfreq, the bandnum is kept constant,
        so it may be possible your centerfrequency changes.

        Parameters
        ----------
        bandnum : int
            Band order number relating to reference frequency.
            If center frequency equals reference frequency, bandnum is 0. 
        fraction : scalar
            1/fraction Octave band. 
            E.g. if fraction is 3, third octave bands are generated.
        referencefreq : scalar
            The reference frequency where the octaves relate to.
        base : {10, 2}
            The base for freqiencies calculations.
            Only base=10 complies with the standard.

        """                     
        self._bandnum = bandnum
        self._referencefreq = referencefreq
        super().__init__(
                fraction=fraction, 
                base=base,
                **kwargs)
        self._update()      

    def _update(self):
        """Refresh instance states."""
        # print('Update', 'FractionalOctaveModel')
        if self._bandnum is None and self._centerfreq:
            self.bandnum = self.frequency_to_bandnum(self._centerfreq)
        if self._bandnum is None:
            return
        self._centerfreq = self.bandnum_to_frequency(self.bandnum)
        super()._update()

    @property
    def bandnum(self):
        """Returns the band number relating to referencereq band with bandnum=0."""
        return self._bandnum

    @bandnum.setter
    def bandnum(self, value):
        """Changes the bandnum value."""
        if value is None:
            return
        if not np.isclose(value, int(value)):
            raise ValueError('Given bandnum value must be integer.')
        self._bandnum = int(value)
        self._update()

    @property
    def referencefreq(self):
        """Returns the reference frequency where bandnum=0."""
        return self._referencefreq

    @referencefreq.setter
    def referencefreq(self, value):
        """Changes the reference frequency"""
        if value <= 0:
            raise ValueError('Given frequency must be greater than 0.')
        self._referencefreq = value
        self._centerfreq = self.bandnum_to_frequency(self.bandnum)
        self._update()

    @property
    def centerfreq(self):
        """Returns the center frequency."""
        return self._centerfreq

    @centerfreq.setter
    def centerfreq(self, value):
        """Changes the center frequency. 
        
        Please note, that the bandnum resulting
        by your given centerfreq must be an integer number. If not, your given centerfreq
        will be adjusted to fulfil the model.
        
        """
        bandnum = self.frequency_to_bandnum(value)
        bandnum_int = int(bandnum)
        centerfreq = self.bandnum_to_frequency(bandnum_int)
        self.bandnum = bandnum_int
        diff = bandnum - bandnum_int
        if not np.isclose(diff, 0):
            warning = (
                'Your given frequency does not correspond to an integer bandnum.' + 
                'The center frequency was changed to fc={} with bandnum={}. '.format( 
                    centerfreq, bandnum_int) +
                'Use `FractionalOctaveBandBaseModel` for custom octave frequencies.')
            warnings.warn(warning)

    @property
    def _get_properties(self):
        """Returns prperties e.g. to create new instance."""
        return dict(
            bandnum=self.bandnum, 
            fraction=self.fraction, 
            referencefreq=self.referencefreq, 
            base=self.base)

    def shift_by(self, bandoffset):
        """Returns an instance shifted by given number of bands."""
        properties = self._get_properties
        properties['bandnum'] += bandoffset
        return self.__class__(**properties)

    def as_bandnum(self, bandnum):
        """Returns an instance at given bandum."""
        properties = self._get_properties
        properties['bandnum'] = bandnum
        return self.__class__(**properties)

    def range_by_centerfreq(self, start, stop):
        """Yields a range of instances for given centerfreq range."""
        start = int(self.frequency_to_bandnum(start))
        stop = int(self.frequency_to_bandnum(stop)) + 1
        yield from self.range_by_bandnum(start, stop, relative=False)

    def range_by_bandnum(self, start, stop, step=None, relative=False):
        """Yields a range of instances for given band number range.
        
        If relative is True, your range will be relative to the bandum
        to the origin instance.

        """
        if step is None:
            step = 1
            
        for index in range(start, stop, step):
            if relative:
                yield self.shift_by(index)
            else:
                yield self.as_bandnum(index)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.as_bandnum(key)
        elif isinstance(key, slice):
            return self.range_by_bandnum(
                key.start, 
                key.stop, 
                key.step, 
                relative=False)       
        
    def bandnum_to_frequency(self, bandnum):
        """Returns frequency from given bandnum."""
        if self._base == 2:
            return self._referencefreq * 2.0**(bandnum/self._fraction)
        elif self._base == 10:
            return self._referencefreq * 10.0**(0.3 * bandnum/self._fraction)
        
    def frequency_to_bandnum(self, centerfreq):
        """Returns bandnum from given frequency."""
        if self._base == 2:
            bandnum = self._fraction * log2(centerfreq/self._referencefreq)
        elif self._base == 10:
            bandnum = (self._fraction / 0.3) * log10(centerfreq/self._referencefreq)
        return bandnum
        

    def __repr__(self):
        return '<{}: #{} 1/{} Oct.@{:.1f} Hz ({:.1f} to {:.1f} Hz)'.format(
            self.__class__.__name__, 
            self.bandnum,
            self.fraction, 
            self.centerfreq, 
            self.loweredgefreq,
            self.upperedgefreq
            )

    def find_nominal_centerfreq(self):
        """Returns corresponding nominal center frequency. 
        
        Only for fraction=1 or fraction=3.

        """
        if self.fraction not in [1, 3]:
            raise ValueError('Can only provide nominal frequencies for fraction 1 and 3.')

        factor = 10**np.floor(log10(self.centerfreq)-2)
        nominalfreqs = factor*STANDARDIZED_NOMINALFREQS
        dist = np.sqrt((nominalfreqs - self.centerfreq)**2)
        found_nominal_centerfreq = nominalfreqs[np.argmin(dist)]

        isgreater = self.centerfreq > np.max(nominalfreqs) 
        islower = self.centerfreq < np.min(nominalfreqs) 
        isoutside = isgreater or islower

        diff = self.centerfreq-found_nominal_centerfreq          
        if isoutside:
            warnings.warn(
                'The centerfrequency is outside the `STANDARDIZED_NOMINALFREQS` list.'
                'Difference is between actual and nominal is {}'.format(diff))
        print(np.abs(diff), self.bandwidth)
        if np.abs(diff) > self.bandwidth*0.5:
            warnings.warn(
                'Difference to actual centerfreq is greater than your half bandwidth.')
        return found_nominal_centerfreq


class BaseFractionalOctaveBandFilter(BaseFractionalOctaveModel):

    def __init__(self, 
                 order=None, 
                 centerfreq=None, 
                 fraction=DEFAULT_FRACTION, 
                 samplerate=1.0, 
                 base=DEFAULT_OCTAVE_BASE, 
                 design_sos_func=None, 
                 **kwargs):
        self._order = order
        self._samplerate = samplerate
        self._sos = None
        self._lastparams = None
        self._design_sos_func = design_sos_func

        if design_sos_func is None:
            def default_design_sos_func(order, lef, uef):
                return butterworth.design_sos('bandpass', order, lef, uef)
            self._design_sos_func = default_design_sos_func

        super().__init__(
            centerfreq=centerfreq,
            fraction=fraction,
            base=base,
            **kwargs)
        self._update()

    def _params(self):
        """Returns params needed to design filter."""
        if self.loweredgefreq and self.upperedgefreq:
            return self._order, self.loweredgefreq/self._samplerate, self.upperedgefreq/self._samplerate

    def _update(self):
        """Refresh instance states."""
        lastparams = self._lastparams
        super()._update()
        params = self._params()
        if params[-1] > 0.5:
            raise ValueError(
                'Upper edge frequency {} must be smaller nyquist frequency {}.'.format(
                    self.upperedgefreq, self.samplerate/2
                ))
        if params is not lastparams:
            self._sos = self._design_sos_func(*params)
            self._lastparams = params

    @property
    def samplerate(self):
        """Returns the samplerate."""
        return self._samplerate

    @samplerate.setter
    def samplerate(self, value):
        """Changes the samplerate property."""
        self._samplerate = float(value)
        self._update()

    def get_maxcenterfreq(self):
        """Returns the max centerfreq for given samplerate."""
        half_sr = self.samplerate/2
        return half_sr - half_sr*self.relative_bandwidth/2

    @property
    def effective_filter_length_estimate(self):
        """Returns an estimate of the effective filter length"""
        return int((self.samplerate*3)//self.bandwidth)

    @property
    def sos(self):
        """Returns the second order section coefficients."""
        return self._sos
    
    @property
    def design_sos_func(self):
        return self._design_sos_func

    @design_sos_func.setter
    def design_sos_func(self, value):
        """Changes the filter design function.
        Must be a function with positional arguments:
        func(order, critical_frequency_1, critical_frequency_2)."""
        self._design_sos_func = value
        self._lastparams = None
        self._update()

    @property
    def order(self):
        """Returns the filter order."""
        return self._order

    @order.setter
    def order(self, value):
        """Changes the filter order."""
        if value % 2 != 0:
            raise ValueError("`order` must be a multiple of 2 for biquads.")

        if isinstance(value, int) :
            self._order = value
            self._update()
        else:
            raise ValueError("`order` must be of type integer.")

    def freqzplot(self, 
                  numsamples=None, 
                  phase=True,
                  axs=None, 
                  relfreq=False,
                  ffilt=False,
                  maxnumsamples=2**21):
        """Creates frequency response plot."""
        from pylab import plt, linspace
        if phase:
            if axs is None:
                _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            else:
                ax1, ax2, ax3 = axs
        elif axs is None:
            ax1 = plt.gca()
        else:
            ax1 = axs
        freq, Y = self.freqz(numsamples, ffilt=ffilt, maxnumsamples=maxnumsamples)
        if relfreq:
            freq /= self.centerfreq
        ax1.semilogx(freq, 20*np.log10(np.abs(Y)))
        ax1.set_ylim([-80, 1])
        ax1.set_ylabel('Attenuation /dB(FS)')
        ax1.grid(True)
        ax1.set_xlim([0, freq[-1]])
        if phase:
            ax1.set_xticklabels([])
            angle = np.unwrap(np.angle(Y))
            ax2.semilogx(freq, angle, '-')
            ax2.set_ylabel('Angle /rad')
            ax2.set_xticklabels([])
            ax2.grid(True)
            ax3.semilogx(freq[:-1], -np.diff(angle)/np.diff(freq))
            ax3.set_ylabel('Groupdelay /s')
            ax3.grid(True)
            ax3.set_xlabel('Frequency')
            plt.axes(ax1)
            return ax1, ax2, ax3
        ax1.set_xlabel('Frequency')
        return ax1

    def freqz(self, numsamples=None, ffilt=True, maxnumsamples=None):
        """Returns frequency response. (freq, Y)."""
        if numsamples is None:
            numsamples = 6 * self.effective_filter_length_estimate
            if maxnumsamples is not None:
                numsamples = min(numsamples, maxnumsamples)

        x = np.zeros(numsamples)
        x[0] = 1.0
        y, _ = self.filter(x, ffilt=ffilt)
        Y = np.fft.rfft(y)
        freq = np.fft.rfftfreq(len(x), 1/self.samplerate)
        return freq, Y

    def filter(self, x, ffilt=False, states=None, axis=0):
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

        if not ffilt:
            y, states = sosfilter_py(
                x, 
                self.sos, 
                states=states, 
                axis=axis)
        elif ffilt:
            y, states = sosfilter_py(
                np.flip(x, axis), 
                self.sos, 
                states=states, 
                axis=axis)
            y, states = sosfilter_py(
                np.flip(y, axis), 
                self.sos, 
                states=states, 
                axis=axis)
        return y, states


class FractionalOctaveBandFilter(
        FractionalOctaveModel,
        BaseFractionalOctaveBandFilter,
        ):

    def get_maxbandnum(self):
        """Returns the maximum bandnum dependend on saplerate"""
        return int(self.frequency_to_bandnum(
            self.get_maxcenterfreq()) - 0.5)

    @property
    def _get_properties(self):
        return dict(
            order=self.order,
            samplerate=self.samplerate,
            centerfreq=self.centerfreq,
            bandnum=self.bandnum, 
            fraction=self.fraction, 
            referencefreq=self.referencefreq, 
            base=self.base)


class BaseFractionalOctaveFilterBank:
    def __init__(self, filterliss=None):
        self._filterlist = filterlist
        self._numbands = len(filters)

    @property
    def filterlist(self):
        return self._filterlist

    @filterlist.setter
    def filterlist(self, value):
        self._filterlist = list(set(list(value)))
        # TODO: check filterlist elements types

    def analyze(self, x, states=None, y=None):
        if y is None:
            y = np.zeros((*x.shape, self._numbands))
        if states is None:
            states = [None] * self._numbands
        for i, band in enumerate(self._filterset):
            y[:, ..., i], states[i] = band.filter(x, states[i])

        return y, states

    def synthesize(self, y):
        pass

    def freqzplot(self):
        for band in self._filterset:
            band.freqzplot()


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
            M[cc, kk] = 2.0/(self.nfft/2+1)
            if kk[0] == 0:
                M[cc, kk[0]] = 1.0/(self.nfft/2+1)

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
