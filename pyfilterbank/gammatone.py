"""This module implements gammatone filters and a filtering routine.

A filterbank is coming soon [Hohmann2002]_.

.. plot::

    import gammatone
    gammatone.example()


TODO:
    - Tests,
    - nice introduction with example,
    - implementing the filterbank class

References
----------

.. [Hohmann2002]
   Hohmann, V., Frequency analysis and synthesis using a Gammatone filterbank,
   Acta Acustica, Vol 88 (2002), 433--442


Functions
---------
"""


from numpy import (array, pi, cos, exp, log, ones_like, sqrt, zeros)
from scipy.misc import factorial
from scipy.signal import lfilter


def equivalent_rectangular_bandwidth(center_frequency):
    """Retrurns equivalent rectangular band width of an auditory filter.

    Parameters
    ----------
    center_frequency : scalar /Hz
        The center frequency in Hertz of the
        desired auditory filter.

    Returns
    -------
    erb : scalar
        Equivalent rectangular bandwidth of
        an auditory filter at `center_frequency`.

    """
    return 24.7 + center_frequency * 0.107933081


def equivalent_rectangular_band_count(center_frequency):
    """Returns the equivalent rectangular band count below center_frequency.

    Parameters
    ----------
    center_frequency : scalar /Hz
        The center frequency in Hertz of the
        desired auditory filter.

    Returns
    -------
    erb_count : scalar
        Number of equivalent bandwidths below `center_frequency`.

    """
    return 21.4 * log10(4.37 * 0.001 * center_frequency + 1)


def design_gammatone_filter(
        sample_rate=44100,
        order=4,
        center_frequency=1000.0,
        band_width=None,
        band_width_factor=1.0,
        attenuation_half_bandwidth_db=-3):
    """Returns filter coefficient of a gammatone filter
    [Hohmann2002]_.

    Parameters
    ----------
    sample_rate : scalar
    order : int
    center_frequency : scalar
    band_width : scalar
    band_width_factor : scalar
    attenuation_half_bandwidth_db : scalar

    Returns
    -------
    b, a : ndarray, ndarray

    """
    if band_width:
        phi = pi * band_width / sample_rate
        alpha = 10**(0.1 * attenuation_half_bandwidth_db / order)
        p = (-2 + 2 * alpha * cos(phi)) / (1 - alpha)
        lambda_ = -p/2 - sqrt(p*p/4 - 1)

    elif band_width_factor:
        erb_audiological = (band_width_factor *
            equivalent_rectangular_bandwidth(center_frequency))
        a_gamma = (pi * factorial(2*order - 2) *
            2**(-(2*order - 2) / factorial(order - 1)**2))
        b = erb_audiological / a_gamma
        lambda_ = exp(-2 * pi * b / sample_rate)
    else:
        raise ValueError(
            'You need to specify either band_width or band_width_factor!')

    beta = 2*pi * center_frequency / sample_rate
    coef = lambda_ * exp(1j*beta)
    factor = 2 * (1 - abs(coef))**order
    b, a = array([factor]), array([1., -coef])
    return b, a


def fosfilter(b, a, order, signal, states=None):
    """Return signal filtered with `b` and `a` (first order section)
    by filtering the signal `order` times.

    This Function was created for filtering signals by first order stion
    cascaded complex gammatone filters.

    Parameters
    ----------
    b, a : ndarray, ndarray
        Filter coefficients of a first order section filter.
        Can be complex valued.
    order : int
        Order of the filter to be applied. This will
        be the count of refiltering the signal order times
        with the given coefficients.
    signal : ndarray
        Input signal to be filtered.
    states : ndarray, default None
        Array with the filter states of length `order`.
        Initial you can set it to None.

    Returns
    -------
    signal : ndarraa
        Output signal, that is filtered and complex valued
        (analytical signal).
    states : ndarray
        Array with the filter states of length `order`.
        You need to loop it back into this function when block
        processing.

    """
    if not states: states = zeros(order) + 0j

    for i in range(order):
        state = [states[i]]
        [signal, state] = lfilter(b, a, signal, zi=state)
        states[i] = state[0]
        b = ones_like(b)
    return signal, states


def example():
    from pylab import plt, np
    sample_rate = 44100
    order = 4
    b, a = design_gammatone_filter(
        sample_rate=sample_rate,
        order=order,
        center_frequency=1000.0,
        attenuation_half_bandwidth_db=-3,
        band_width=100.0)

    x = zeros(10000)
    x[0] = 1.0
    y, states = fosfilter(b, a, order, x)
    y = y[:800]
    plt.plot(np.real(y), label='Re(z)')
    plt.plot(np.imag(y), label='Im(z)')
    plt.plot(np.abs(y), label='|z|')
    plt.legend()
    plt.show()
    return y


if __name__ == '__main__':
    y = example()
