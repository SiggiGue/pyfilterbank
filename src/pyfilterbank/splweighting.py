"""This module implements spectral weighting filters for the sound pressure level (SPL)
in air according to [IEC-61672]_. Spectral weighting is part of aucoustic measurements.
It is used by sound level meters for example. The weighting functions are derived
from different equal loudness contours of human hearing. The weighted levels aim to
provide a better correlation to the perception of loudness.


Implemented weighting functions
-------------------------------

There are three weighting functions implemented:

    * A-Weighting: based on the 40-phon equal loudness contour
    * B- and C-weighting: for sounds above 70 phon,
      (B-Weighting is not used that often)

The filter coefficient design is based on the implementation of A- and C-weighting in [2]_.

The weighting functions are defined in [IEC-61672]_ can be described
by the following equations:

.. math::

   R_A (f) = \\frac{12200^2 f^4}
             {(f^2+20.6^2)(f^2+12200^2)\\sqrt{(f^2+107.7.5)^2}\\sqrt{(f^2+737.9^2)}}

   R_B (f) = \\frac{12200^2 f^3}
             {(f^2+20.6^2)(f^2+12200^2)\\sqrt{(f^2+158.5^2)}}

   R_C (f) = \\frac{12200^2 f^2}
             {(f^2+20.6^2)(f^2+12200^2)}


The frequency responses absolute values of all implemented weighting filters can be seen in the following figure:

.. plot::

   import splweighting
   fig, ax = splweighting.plot_weightings()
   fig.show()


References
----------

.. [IEC-61672] Electroacoustics - Sound Level Meters (http://www.iec.ch)
.. [2] *Christophe Couvreur*, MATLAB(R) implementation of weightings,
   http://www.mathworks.com/matlabcentral/fileexchange/69-octave,
   Faculte Polytechnique de Mons (Belgium) couvreur@thor.fpms.ac.be


Functions
---------
"""

from numpy import pi, convolve
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter


def weight_signal(data, sample_rate=44100, weighting='A'):
    """Returns filtered signal with a weighting filter.

    Parameters
    ----------
    data : ndarray
        Input signal to be filtered.
    sample_rate : int
        Sample rate of the signal.
    weighting : {'A', 'B', 'C'}
        Specify the weighting function by a string.

    Returns
    -------
    outdata : ndarray
        Filtered output signal. The output will be weighted by
        the specified filter function.
    """
    b, a = _weighting_coeff_design_funsd[weighting](sample_rate)
    return lfilter(b, a, data)

def a_weighting_coeffs_design(sample_rate):
    """Returns b and a coeff of a A-weighting filter.

    Parameters
    ----------
    sample_rate : scalar
        Sample rate of the signals that well be filtered.

    Returns
    -------
    b, a : ndarray
        Filter coefficients for a digital weighting filter.

    Examples
    --------
    >>> b, a = a_weighting_coeff_design(sample_rate)

    To Filter a signal use scipy lfilter:

    >>> from scipy.signal import lfilter
    >>> y = lfilter(b, a, x)

    See Also
    --------
    b_weighting_coeffs_design : B-Weighting coefficients.
    c_weighting_coeffs_design : C-Weighting coefficients.
    weight_signal : Apply a weighting filter to a signal.
    scipy.lfilter : Filtering signal with `b` and `a` coefficients.
    """

    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    numerators = [(2*pi*f4)**2 * (10**(A1000 / 20.0)), 0., 0., 0., 0.];
    denominators = convolve(
        [1., +4*pi * f4, (2*pi * f4)**2],
        [1., +4*pi * f1, (2*pi * f1)**2]
    )
    denominators = convolve(
        convolve(denominators, [1., 2*pi * f3]),
        [1., 2*pi * f2]
    )
    return bilinear(numerators, denominators, sample_rate)

def b_weighting_coeffs_design(sample_rate):
    """Returns `b` and `a` coeff of a B-weighting filter.

    B-Weighting is no longer described in DIN61672.

    Parameters
    ----------
    sample_rate : scalar
        Sample rate of the signals that well be filtered.

    Returns
    -------
    b, a : ndarray
        Filter coefficients for a digital weighting filter.

    Examples
    --------
    >>> b, a = b_weighting_coeff_design(sample_rate)

    To Filter a signal use :function: scipy.lfilter:

    >>> from scipy.signal import lfilter
    >>> y = lfilter(b, a, x)

    See Also
    --------
    a_weighting_coeffs_design : A-Weighting coefficients.
    c_weighting_coeffs_design : C-Weighting coefficients.
    weight_signal : Apply a weighting filter to a signal.

    """

    f1 = 20.598997
    f2 = 158.5
    f4 = 12194.217
    B1000 = 0.17
    numerators = [(2*pi*f4)**2 * (10**(B1000 / 20)), 0, 0, 0];
    denominators = convolve(
        [1, +4*pi * f4, (2*pi * f4)**2],
        [1, +4*pi * f1, (2*pi * f1)**2]
    )
    denominators = convolve(denominators, [1, 2*pi * f2])
    return bilinear(numerators, denominators, sample_rate)


def c_weighting_coeffs_design(sample_rate):
    """Returns b and a coeff of a C-weighting filter.

    Parameters
    ----------
    sample_rate : scalar
        Sample rate of the signals that well be filtered.

    Returns
    -------
    b, a : ndarray
        Filter coefficients for a digital weighting filter.

    Examples
    --------
    b, a = c_weighting_coeffs_design(sample_rate)

    To Filter a signal use scipy lfilter:

    from scipy.signal import lfilter
    y = lfilter(b, a, x)

    See Also
    --------
    a_weighting_coeffs_design : A-Weighting coefficients.
    b_weighting_coeffs_design : B-Weighting coefficients.
    weight_signal : Apply a weighting filter to a signal.

    """

    f1 = 20.598997
    f4 = 12194.217
    C1000 = 0.0619
    numerators = [(2*pi * f4)**2 * (10**(C1000 / 20)), 0, 0]
    denominators = convolve(
        [1, +4*pi * f4, (2*pi * f4)**2],
        [1, +4*pi * f1, (2*pi * f1)**2]
    )
    return bilinear(numerators, denominators, sample_rate)


# This dictionary should contain all labels and functions
# for weighting coeff design functions:
_weighting_coeff_design_funsd = {
    'A': a_weighting_coeffs_design,
    'B': b_weighting_coeffs_design,
    'C': c_weighting_coeffs_design
}

def plot_weightings():
    """Plots all weighting functions defined in :module: splweighting."""
    from scipy.signal import freqz
    from pylab import plt, np

    sample_rate = 48000
    num_samples = 2*4096

    fig, ax = plt.subplots()

    for name, weight_design in sorted(
            _weighting_coeff_design_funsd.items()):
        b, a = weight_design(sample_rate)
        w, H = freqz(b, a, worN=num_samples)

        freq = w*sample_rate / (2*np.pi)

        ax.semilogx(freq, 20*np.log10(np.abs(H)+1e-20),
                    label='{}-Weighting'.format(name))

    plt.legend(loc='lower right')
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Damping / dB')
    plt.grid(True)
    plt.axis([10, 20000, -80, 5])
    return fig, ax


if __name__ == '__main__':
    fig, ax = plot_weightings()
    fig.show()
