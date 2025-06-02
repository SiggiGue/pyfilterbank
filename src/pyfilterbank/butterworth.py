"""The :mod:`butterworth` module provides functions to design butterwort filters.


"""
from numpy import (mod, exp, zeros, ones, arange, kron, real, flipud,
                   conj, pi, fliplr, sqrt, tan, tile, concatenate, append, double)
from scipy.signal import butter, buttord, buttap

from pyfilterbank.sosfiltering import bilinear_sos

lowpass = 'lowpass'
highpass = 'highpass'
bandpass = 'bandpass'
bandstop = 'bandstop'

def butter_sos(band, L, v1, v2=0.0):
    """Compute weights of a digital Butterworth filter in cascade form.

    Parameters
    ----------
    band : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    L : int
        Number of lowpass poles. L is doubled for
        'bandpass' and 'bandstop'. L must be even.
    v1 : scalar
        First critical frequency (Hz-s); 0.0 <v1 < 0.5.
    v2 : scalar
        Second critical frequency; v1 < v2 < 0.5.
        v2 is used only if 'bandpass' or 'bandstop'.

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

    # Check for errors;
    if(v1<=0 or v1>=.5):
       raise Exception('Argument v1 must be >0.0 and <0.5')
    elif v2!=0 and (v2<=v1 or v2>=0.5):
       raise Exception('Argument v2 must be >v1 and <0.5')

   # Get the anlalog weights and convert to digital.
    if(v2 == 0):
       d, c = butter_analog_sos(band, L, tan(pi*v1))
    else:
       d, c = butter_analog_sos(band, L, tan(pi*v1), tan(pi*v2))
    b, a = bilinear_sos(d, c)
    # TODO: cut this step and give back b, a
    # but the other functions needing an sosmat should be adapted.
    sosmat = flipud(concatenate((b, a), axis=1))
    return sosmat


def butter_analog_sos(band, L, w1, w2=0):
    """Returns analog filter coeffitients for Butterworth filters.
    compute analog weights of a butterworth filter

    Parameters
    ----------
    band : {'lowpass', 'highpass, 'bandpass', 'bandstop'}
    L : int
        Order of lowpass / highpass filter.
    w1 : scalar
        Critical frequency one.
    w2 : scalar
        Critical frequency two. (for 'bandpass' or 'bandstop').

    Returns
    -------
    d, c :  Analog weights of the filter

    Notes
    -----
    implements SOS H(s)  butterwort
    if you need H(z) apply a bilinear transform

    """
    band = band.lower()
    L2 = int(L / 2.0)

    if mod(L, 2):
        raise Exception('Number of poles L must be even')
    if w1 <= 0:
        raise Exception('Frequency w1 must be in rad/s and >0')

    # define center frequency wc:
    if band==lowpass or band==highpass:
        wc = w1
    else:
        wc = w2-w1

    p = wc * exp(-1j * (2*arange(1,L2+1,dtype=double) + L-1) * pi / (2*L))

    # defining the lowpass filter:
    d = zeros((L2, 2), dtype=double).astype(complex)
    c = ones((L2, 2), dtype=double).astype(complex)
    d[:,1] = wc * ones(L2, dtype=double)
    c[:,1] = -p

    if band == highpass:
        d[:, 0] = d[:, 1]
        d[:, 1] = 0.0
        c = fliplr(c)
        c[:, 1] = c[:, 1] * wc**2

    elif band == bandpass:
        d[:, 0] = d[:, 1]
        d[:, 1] = zeros(L2)
        d = append(d, d, axis=0)
        d[L2:, 0] = zeros(L2)
        d[L2:, 1] = ones(L2)
        root = sqrt(c[:, 1]**2 - 4*c[:, 0]**2 * w1*w2)
        r1 = (-c[:, 1] + root) / (2 * c[:, 0])
        r2 = (-c[:, 1] - root) / (2 * c[:, 0])
        c[:, 0] = c[:, 0]
        c[:, 1] = -c[:, 0] * r1
        c = append(c, c, axis=0)
        c[L2:, 0] = 1.0
        c[L2:, 1] = -r2

    elif band == bandstop:
        root = sqrt(d[:, 0]**2 * wc**4 - 4*d[:, 1]**2 * w1*w2)
        r1 = (-d[:, 0] * wc**2 + root) / (2 * d[:, 1])
        r2 = (-d[:, 0] * wc**2 - root) / (2 * d[:, 1])
        d[:, 0] = d[:L2, 1]
        d[:, 1] = -d[:L2, 1] * r1
        d = append(d, d, axis=0)
        d[L2:2*L2, 0] = ones(L2)
        d[L2:2*L2, 1] = -r2
        root = sqrt(c[:, 0]**2 * wc**4 - 4*c[:, 1]**2 * w1*w2)
        r1 = (-c[:, 0] * wc**2 + root) / (2 * c[:, 1])
        r2 = (-c[:, 0] * wc**2 - root) / (2 * c[:, 1])
        c[:, 0] = c[:L2, 1]
        c[:, 1] = -c[:L2, 1] * r1
        c = append(c, c, axis=0)
        c[L2:2*L2, 0] = ones(L2)
        c[L2:2*L2, 1] = -r2

    return d, c
