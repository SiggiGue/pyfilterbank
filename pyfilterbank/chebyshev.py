import numpy as np
from .sosfiltering import bilinear_sos



def design_sos(band, order, freq1, freq2=0.0, stopband_gain_db=-60):
    """Returns weights of a digital Chebyshev filter in cascaded biquad form.

    Parameters
    ----------
    band : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    order : int
        Number of lowpass poles. `order` is doubled for
        'bandpass' and 'bandstop'. `order` must be even.
    freq1 : scalar
        First critical frequency; 0.0 <freq1 < 0.5.
    freq2 : scalar
        Second critical frequency; freq1 < freq2 < 0.5.
        freq2 is used only if 'bandpass' or 'bandstop'.
    stopband_gain_db : scalar
        Gain in dB for stopband. Default -60 dB.

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
    isinvalid_critfreq_2 = (
        freq2 <= freq1 or freq2 >= 0.5)
    # Check for correct input;
    if freq1 <= 0 or freq1 >= 0.5:
        raise ValueError('Argument `freq1` must be > 0.0 and < 0.5')
    elif freq2 > 0 and isinvalid_critfreq_2:
        raise Exception('Argument `freq2` must be > `freq1` and < 0.5')

    # Get the anlalog weights and convert to digital.
    d, c = design_analog_sos(
        band=band, 
        order=order, 
        freq1=np.tan(np.pi*freq1), 
        freq2=np.tan(np.pi*freq2),
        stopband_gain_db=stopband_gain_db)

    b, a = bilinear_sos(d, c)
    sosmat = np.flipud(np.concatenate((b, a), axis=1))
    return sosmat


def design_analog_sos(band, order, freq1, freq2=0, stopband_gain_db=-60):
    """Returns analog filter weights for Chebyshev filters.

    Parameters
    ----------
    band : {'lowpass', 'highpass, 'bandpass', 'bandstop'}
    order : int
        Order of lowpass / highpass filter.
    freq1 : scalar
        Critical frequency one.
    freq2 : scalar
        Critical frequency two. (for 'bandpass' or 'bandstop').
    stopband_gain_db : scalar
        Stop band gain in dB, Default -60.
        

    Notes
    -----

    Original docs from Stearns:
    % 
    % Chebyshev analog lowpass, highpass, bandpass, or bandstop weights.
    %
    % Arrays d and c are numerator and denominator weights
    % of the analog filter in cascade form using single-pole sections 
    % H(1,s),H(2,s),..., and H(L/2,s). Thus d and c are L/2 x 2 arrays,
    % and H(s)=H(1,s)H(1,s)'...H(L/2,s)H(L/2,s)'.
    % 
    % Inputs: band =1(lowpass) 2(highpass) 3(bandpass) or 4(bandstop).
    %         L    =# Lowpass poles. L must be even in this function.
    %         dB   =stopband gain in dB; for example, "-60".
    %         freq1   =lower critical frequency in rad/s.
    %         freq2   =upper critical frequency (rad/s). Required only
    %               if band = 2 or 3.
    %
    % Outputs: d =L/2 x 2 array of numerator weights.
    %          c =L/2 x 2 array of denominator weights.
    % Note: If band =3 or 4, L is doubled in the translation.
    """
    if not isinstance(band, str):
        raise ValueError('band` must be of type str not {}'.format(type(band)))

    band = band.lower()
    half_order = int(order / 2.0)

    if np.mod(order, 2):
        raise ValueError('Number of lowpass poles `order` must be even.')
    elif stopband_gain_db > -10:
        raise ValueError('stoppand_gain_db must be -10 or less.')
    elif freq1 <= 0:
        raise ValueError('Frequency freq1 must be in rad/s and >0')

    # define critical frequency 0
    if band == 'lowpass' or band == 'highpass':
        wc = freq1
    else:
        wc = freq2 - freq1

    # Compute and test ws.
    ws = wc * np.cosh(
        np.arccosh(np.sqrt(10**(-stopband_gain_db/10)-1)) / order)
    if ws <= wc:
        raise ValueError('Design won''t work. Please increase either order or stopband_gain_db.')

    # Basic lowpass design
    # Poles (column).
    zeta = 1.0 / np.cosh(order * np.arccosh(ws / wc))
    alpha = (1.0 / order) * np.arcsinh(1 / zeta)
    beta = (2 * np.arange(1, half_order + 1) - order - 1) * np.pi / (2 * order)
    sigma = wc * (np.sinh(alpha) * np.cos(beta) + 1j * np.cosh(alpha) * np.sin(beta))
    p = (-wc * ws / sigma)

    # Zeros (column).
    sigma = 1j * wc * np.cos((2 * np.arange(1, half_order+1) - 1) * np.pi / (2*order))
    z = (-wc * ws / sigma)

    d = np.zeros((half_order, 2), dtype=np.double).astype(complex)
    c = np.zeros((half_order, 2), dtype=np.double).astype(complex)

    if band == 'lowpass':
        d[:, 0] = p
        d[:, 1] = -p * z
        c[:, 0] = z
        c[:, 1] = -z * p

    elif band == 'highpass':
        d[:, 0] = 1.0
        d[:, 1] = -wc**2 / z
        c[:, 0] = 1.0
        c[:, 1] = -wc**2 / p

    elif band == 'bandpass':
        rz = (z + 1j * np.sqrt(4*freq1*freq2 - z**2)) / 2.0
        d[:, 0] = p
        d[:, 1] = -rz * p
        rz = (np.conj(z) + 1j * np.sqrt(4*freq1*freq2 - np.conj(z)**2)) / 2.0
        d = np.append(d, d, axis=0)
        d[half_order:, 0] = 1.0
        d[half_order:, 1] = -rz
        rp = (p + 1j * np.sqrt(4*freq1*freq2 - p**2)) / 2.0
        c[:, 0] = z
        c[:, 1] = -rp * z
        rp = (np.conj(p) + 1j * np.sqrt(4*freq1*freq2 - np.conj(p)**2)) / 2.0
        c = np.append(c, c, axis=0)
        c[half_order:, 0] = 1.0
        c[half_order:, 1] = -rp

    elif band == 'bandstop':
        rz = (wc**2 / z + 1j * np.sqrt(4*freq1*freq2-wc**4 / z**2) ) / 2.0
        d[:, 0] = 1.0
        d[:, 1] = -rz
        rz = (wc**2 / np.conj(z) + 1j * np.sqrt(4*freq1*freq2 - wc**4 / np.conj(z)**2)) / 2.0
        d = np.append(d, d, axis=0)
        d[half_order:, 0] = 1.0
        d[half_order:, 1] = -rz
        rp = (wc**2 / p + 1j * np.sqrt(4*freq1*freq2 - wc**4 / p**2)) / 2.0
        c[:, 0] = 1.0
        c[:, 1] = -rp
        rp = (wc**2 / np.conj(p) + 1j * np.sqrt(4*freq1*freq2 - wc**4 / np.conj(p)**2)) / 2.0
        c = np.append(c, c, axis=0)
        c[half_order:, 0] = 1.0
        c[half_order:, 1] = -rp

    else:
        raise ValueError(
            "Argument `band` must be 'lowpass', 'heighpass', 'bandpass' or 'bandstop'"
            " not '{}'".format(band)
        )
    return d, c