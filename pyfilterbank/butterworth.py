""":mod:`butterworth` module provides functions to design butterwort filters.

"""
import numpy as np

from pyfilterbank.sosfiltering import bilinear_sos


def design_sos(band, order, freq1, freq2=0.0):
    """Returns weights of a digital Butterworth filter in cascaded biquad form.

    Parameters
    ----------
    band : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    order : int
        Filter order. `order` is doubled for
        'bandpass' and 'bandstop'. `order` must be even.
    freq1 : scalar
        First critical frequency; 0.0 < freq1 < 0.5.
    freq2 : scalar
        Second critical frequency; freq1 < freq2 < 0.5.
        freq2 is used only if 'bandpass' or 'bandstop'.

    Returns
    -------
    sosmat : ndarray
        Contains the numerator and denominator coeffs for each
        cascade in one row.

    Notes
    -----
    Adapted from: Samuel D. Stearns, "Digital Signal Processing
    with Examples in MATLAB"
    
    """

    # Check for correct input;
    if freq1 <= 0 or freq1 >= 0.5:
        raise ValueError('Argument freq1 must be >0.0 and <0.5')
    elif freq2 != 0 and (freq2 <= freq1 or freq2 >= 0.5):
        raise Exception('Argument freq2 must be >freq1 and <0.5')

    # Get the anlalog weights and convert to digital.
    d, c = design_analog_sos(
        band=band, 
        order=order, 
        freq1=np.tan(np.pi*freq1), 
        freq2=np.tan(np.pi*freq2))

    b, a = bilinear_sos(d, c)
    sosmat = np.ascontiguousarray(np.flipud(np.concatenate((b, a), axis=1)))
    return sosmat


def design_analog_sos(band, order, freq1, freq2=0):
    """Returns analog filter coeffitients for Butterworth filters.
    compute analog weights of a butterworth filter

    Parameters
    ----------
    band : {'lowpass', 'highpass, 'bandpass', 'bandstop'}
    order : int
        Order of lowpass / highpass filter.
    freq1 : scalar
        Critical frequency one.
    freq2 : scalar
        Critical frequency two. (for 'bandpass' or 'bandstop').

    Returns
    -------
    d, c :  Analog weights of the filter

    """
    if not isinstance(band, str):
        raise ValueError('band` must be of type str not {}'.format(type(band)))

    band = band.lower()
    half_order = int(order / 2.0)
    if np.mod(order, 2):
        raise ValueError('Number of poles `order` must be even')
    if freq1 <= 0:
        raise ValueError('Frequency freq1 must be in rad/s and >0')

    # define critical_frequency_0:
    if band == 'lowpass' or band == 'highpass':
        critical_frequency_0 = freq1
    else:
        critical_frequency_0 = freq2 - freq1

    valuerange = 2.0 * np.arange(1, half_order+1, dtype=np.double) + order - 1
    p = critical_frequency_0 * np.exp(-1j * valuerange * np.pi / (2.0 * order))

    # defining the lowpass filter:
    d = np.zeros((half_order, 2), dtype=np.double).astype(complex)
    c = np.ones((half_order, 2), dtype=np.double).astype(complex)
    d[:, 1] = critical_frequency_0 * np.ones(half_order, dtype=np.double)
    c[:, 1] = -p

    if band == 'lowpass':
        return d, c
    
    elif band == 'highpass':
        d[:, 0] = d[:, 1]
        d[:, 1] = 0.0
        c = np.fliplr(c)
        c[:, 1] = c[:, 1] * critical_frequency_0**2

    elif band == 'bandpass':
        d[:, 0] = d[:, 1]
        d[:, 1] = np.zeros(half_order)
        d = np.append(d, d, axis=0)
        d[half_order:, 0] = np.zeros(half_order)
        d[half_order:, 1] = np.ones(half_order)
        root = np.sqrt(c[:, 1]**2 - 4*c[:, 0]**2 * freq1*freq2)
        r1 = (-c[:, 1] + root) / (2 * c[:, 0])
        r2 = (-c[:, 1] - root) / (2 * c[:, 0])
        c[:, 0] = c[:, 0]
        c[:, 1] = -c[:, 0] * r1
        c = np.append(c, c, axis=0)
        c[half_order:, 0] = 1.0
        c[half_order:, 1] = -r2

    elif band == 'bandstop':
        root = np.sqrt(
            d[:, 0]**2 * critical_frequency_0**4 - 
            4*d[:, 1]**2 * freq1*freq2)
        r1 = (-d[:, 0] * critical_frequency_0**2 + root) / (2 * d[:, 1])
        r2 = (-d[:, 0] * critical_frequency_0**2 - root) / (2 * d[:, 1])
        d[:, 0] = d[:half_order, 1]
        d[:, 1] = -d[:half_order, 1] * r1
        d = np.append(d, d, axis=0)
        d[half_order:2*half_order, 0] = np.ones(half_order)
        d[half_order:2*half_order, 1] = -r2
        root = np.sqrt(
            c[:, 0]**2 * critical_frequency_0**4 - 
            4*c[:, 1]**2 * freq1*freq2)
        r1 = (-c[:, 0] * critical_frequency_0**2 + root) / (2 * c[:, 1])
        r2 = (-c[:, 0] * critical_frequency_0**2 - root) / (2 * c[:, 1])
        c[:, 0] = c[:half_order, 1]
        c[:, 1] = -c[:half_order, 1] * r1
        c = np.append(c, c, axis=0)
        c[half_order:2*half_order, 0] = np.ones(half_order)
        c[half_order:2*half_order, 1] = -r2
        
    else:
        raise ValueError(
            "Argument `band` must be 'lowpass', 'heighpass', 'bandpass' or 'bandstop'"
            " not {}".format(band)
        )

    return d, c
