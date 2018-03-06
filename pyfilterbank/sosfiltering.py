# -*- coding: utf-8 -*-
"""This module contains second order section filtering routines
implemented in c, cffi and numpy.

A bilinear transform converting sos analog weights to
sos digital weights is provided by :func:`bilinear_sos`.

There are different implementations of sos filtering routines:
    - A scipy.lfilter()-based :func:`sosfilter_py`
    - cffi implementations with float- and double-precision
      and a mimo-implementation:
          * :func:`sosfilter_c` (float)
          * :func:`sosfilter_double_c` (double)
          * :func:`sosfilter_double_mimo_c`
            (multi channel input and 3-dim output).

    - prototypes for the c-implementations
      (slowest, only for debugging)

The c-implementations are for real valued signals only.

With the function :func:`freqz` you can check the
frequency response of your second order section filters.

For the :mod:`cffi` you need :mod:`pycparser` being installed.

Compiling the c source
----------------------
Firstly i implemented a prototype-function in python
for easy debugging "sosfilter_cprototype_py()".
After that i translated this prototype into a c-function. By
compiling a shared library from it with the listed
steps below, one can use the python cffi to access this
shared library in python. ::
    $ gcc -c -std=c99 -O3 sosfilter.c
    $ gcc -shared -o sosfilter.so sosfilter.o
    $ or the last line for windows users:
    $ gcc -shared -o sosfilter.dll sosfilter.o

Functions
---------
"""
import os
from sys import platform
from platform import architecture

import numpy as np
from cffi import FFI
from scipy.signal import lfilter


ffi = FFI()
ffi.cdef("""
void sosfilter(float*, int, float*, int, float*);
void sosfilter_double(double*, int, double*, int, double*);
void sosfilter_double_mimo(double*, int, int, double*, int, int, double*);
""")

if platform == 'win32' and architecture()[0] == '64bit':
    _dl = 'sosfilt64.dll'
elif platform == 'win32' and architecture()[0] == '32bit':
    _dl = 'sosfilt32.dll'
else:
    _dl = 'sosfilt.so'


if __name__ != '__main__':
    _mylibpath = os.path.join('.', os.path.dirname(__file__))
else:
    _mylibpath = os.curdir

_mylibpath = os.path.join(_mylibpath, _dl)

_c = ffi.dlopen(_mylibpath)


def sosfilter_c(signal, sos, states=None):
    """Second order section filter function using cffi

    signal_out, states = sosfilter_c(signal_in, sos, states=None)

    Parameters
    ----------
    signal : ndarray
        Input array of shape (N x 0).
    sos : ndarray
        Second order section coefficients array of shape (K*6 x 0).
        One biquad -> 6 coefficients:
        :code:`[b00, b01, b02, a00, a01, a02, ..., bK1, ..., aK2]`
    states : ndarray
        Array with filter states. Initial value can be None.

    Returns
    -------
    signal : ndarray
        Filtered signal of shape (N x 0).
    states : ndarray
        Array with filter states.

    """

    signal_c = ffi.new(
        'char[]', np.array(signal, dtype=np.float32).flatten().tostring())
    sos_c = ffi.new(
        'char[]', np.array(sos, dtype=np.float32).flatten().tostring())
    nsamp = int(len(signal))
    ksos = int(sos.size/6)

    if isinstance(states, type(None)):
        states = np.zeros(ksos*2).astype(np.double)

    states_c = ffi.new(
        'char[]', np.array(states, dtype=np.float32).flatten().tostring())

    _c.sosfilter(ffi.cast("float*", signal_c),
                 nsamp,
                 ffi.cast("float*", sos_c),
                 ksos,
                 ffi.cast("float*", states_c))

    out = np.fromstring(
        ffi.buffer(signal_c),
        dtype=np.float32,
        count=nsamp
    )
    states = np.fromstring(
        ffi.buffer(states_c),
        dtype=np.float32,
        count=len(states)
    )
    return out, states


def sosfilter_double_c(signal, sos, states=None):
    """Second order section filter function using cffi, double precision.

    signal_out, states = sosfilter_c(signal_in, sos, states=None)

    Parameters
    ----------
    signal : ndarray
        Signal array of shape (N x 0).
    sos : ndarray
        Second order section coefficients array of shape (K*6 x 0).
        One biquad -> 6 coefficients:
        ``[b00, b01, b02, a00, a01, a02, ..., b10, bK1 ... , aK2]``
    states : ndarray
        Filter states, initial value can be None.

    Returns
    -------
    signal :
        Filtered signal array of shape (N x 0).
    states : ndarray
        Filter states, initial value can be None.

    """

    signal_c = ffi.new(
        'char[]', np.array(signal, dtype=np.double).flatten().tostring())
    sos_c = ffi.new(
        'char[]', np.array(sos, dtype=np.double).flatten().tostring())
    nsamp = int(len(signal))
    ksos = int(sos.size/6)

    if isinstance(states, type(None)):
        states = np.zeros(ksos*2).astype(np.double)

    states_c = ffi.new(
        'char[]', np.array(states, dtype=np.double).flatten().tostring())

    _c.sosfilter_double(ffi.cast("double*", signal_c),
                        nsamp,
                        ffi.cast("double*", sos_c),
                        ksos,
                        ffi.cast("double*", states_c))

    out = np.fromstring(
        ffi.buffer(signal_c),
        dtype=np.double,
        count=nsamp)
    states = np.fromstring(
        ffi.buffer(states_c),
        dtype=np.double,
        count=len(states))
    return out, states


def sosfilter_double_mimo_c(signal, sos, states=None):
    """Second order section filter function for  multi channel input
    using cffi, double precision

    signal_out, states = sosfilter_c(signal_in, sos, states=None)

    Parameters
    ----------
    signal : ndarray
        Signal array of shape (N x C).
    sos : ndarray
        Second order section filter coefficients (K*6 x B x C) np-array.

    states : ndarray
        Filter states, initial can be None.
        Otherwise shape is (K*2 x B x C)

    Returns
    -------
    signal : ndarray
        Filtered signal of shape (N x B x C).
        Where N is the number of samples, B is th number of
        filter bands and C is the number of signal channels.
    states : ndarray
        Filter states of shape (K*2 x B x C).

    """

    shape_signal = signal.shape
    nframes = int(shape_signal[0])
    if len(shape_signal) > 1:
        nchan = int(shape_signal[1])
    else:
        nchan = int(1)

    shape_sos = sos.shape
    ksos = int(shape_sos[0]/6)
    if len(shape_sos) > 1:
        kbands = int(shape_sos[1])
        if len(shape_sos) == 2 and nchan > 1:
            sos = np.tile(sos.flatten('F'), (nchan))
    else:
        kbands = int(1)

    if isinstance(states, type(None)):
        states = np.zeros(nchan*kbands*ksos*2).astype(np.double)

    states_c = ffi.new(
        'char[]', np.array(states, dtype=np.double).flatten('F').tostring())

    sos_c = ffi.new(
        'char[]', np.array(sos, dtype=np.double).flatten('F').tostring())

    if nchan > 1:
        signal = np.tile(signal, (kbands, 1))
    else:
        signal = np.tile(signal, (kbands))

    shape_signal = signal.shape
    signal_c = ffi.new(
        'char[]', np.array(signal, dtype=np.double).T.flatten().tostring())

    _c.sosfilter_double_mimo(
        ffi.cast("double*", signal_c),
        nframes,
        nchan,
        ffi.cast("double*", sos_c),
        ksos,
        kbands,
        ffi.cast("double*", states_c)
    )

    out = np.fromstring(
        ffi.buffer(signal_c),
        dtype=np.double,
        count=signal.size
    )
    states = np.fromstring(
        ffi.buffer(states_c),
        dtype=np.double,
        count=len(states)
    )
    return out.reshape((nframes, kbands, nchan), order='F'), states


def sosfilter_mimo_cprototype_py(signal_in, sos_in, states_in=None):
    """Prototype for the mimo c-filter function.
    Implements a IIR DF-II biquad filter strucure. But with multiple
    input und multiple bands."""
    signal = signal_in.copy().flatten('F')
    shape_signal = signal_in.shape
    print(len(signal))
    nframes = int(signal_in.shape[0])
    print(nframes)
    nchan = int(signal_in.shape[2])
    print(nchan)
    sos = np.tile(sos_in.copy().flatten('F'), (nchan))
    ksos = int(sos_in.shape[0]/6)
    print(ksos)
    kbands = int(sos_in.shape[1])
    print(kbands)
    if not states_in:
        states = np.zeros(nchan*ksos*kbands*2)
    else:
        states = states_in

    ii = 0
    for c in range(nchan):
        for b in range(kbands):
            for k in range(ksos):
                w1 = states[c*ksos*kbands*2 + b*ksos*2 + k*2]
                w2 = states[c*ksos*kbands*2 + b*ksos*2 + k*2 + 1]
                b0 = sos[ii]
                ii += 1
                b1 = sos[ii]
                ii += 1
                b2 = sos[ii]
                ii += 1
                a0 = sos[ii]
                ii += 1
                a1 = sos[ii]
                ii += 1
                a2 = sos[ii]
                ii += 1

                for n in range(nframes):
                    w0 = signal[c*nframes*kbands + b*nframes + n]
                    w0 = w0 - a1*w1 - a2*w2
                    yn = b0*w0 + b1*w1 + b2*w2
                    w2 = w1
                    w1 = w0
                    signal[c*nframes*kbands + b*nframes + n] = yn

            states[c*ksos*kbands*2 + b*ksos*2 + k*2] = w1
            states[c*ksos*kbands*2 + b*ksos*2 + k*2 + 1] = w2
    return signal.reshape(shape_signal), states


def sosfilter_cprototype_py(signal, sos, states):
    """Prototype for second order section filtering c function.
    Implements a IIR DF-II biquad filter strucure.
    """
    N = int(len(signal))
    K = int(sos.size/6)
    if isinstance(states, type(None)):
        states = np.zeros(K*2).astype(np.double)
    signal = signal.copy()  # only python specific
    sos = sos.copy().flatten()
    yn = 0.0  # buffer for output
    w0 = 0.0  # signal states

    for k in range(K):
        # get coefficients of current biquad
        w1 = states[k*2]
        w2 = states[k*2+1]
        b0 = sos[k*6]
        b1 = sos[k*6+1]
        b2 = sos[k*6+2]
        a0 = sos[k*6+3]
        a1 = sos[k*6+4]
        a2 = sos[k*6+5]

        for n in range(N):
            # get a sample
            w0 = signal[n].copy()
            # recursive path
            w0 = w0 - a1*w1 - a2*w2
            # transversal path
            yn = b0*w0 + b1*w1 + b2*w2
            # delays
            w2 = w1
            w1 = w0
            # write output signal
            signal[n] = yn

    states[k*2] = w1
    states[k*2+1] = w2

    return signal, states


def sosfilter_py(x, sos, states=None):
    """Second order section filter routing with scipy lfilter.

    Parameters
    ----------
    x : ndarray
        Input signal array.
    sos : ndarray
        Second order section coefficients array.
    states : ndarray or None
        Filter states, initial value can be None.


    Returns
    -------
    signal : ndarray
        Filtered signal.
    states : ndarray
        Array with filter states.

    """
    n = sos.shape[0]
    if isinstance(states, type(None)):
        states = dict()
        for i in np.arange(n):
            states[i] = np.zeros(2)
    for ii in np.arange(n):
        zi = states[ii]
        b = sos[ii, :3]
        a = sos[ii, 3:]
        x, zi = lfilter(b, a, x, 0, zi=zi)
        states[ii] = zi
    return x, states


def bilinear_sos(d, c):
    """Bilinear transformation of analog weights to digital weights.
    >>>>>>> 8d01abb1e1f252834c0666d50c645dd3d35a1f52

    Bilinear transformation of analog weights to digital weights.
    Weights of IIR digital filter in cascade form with
    2-pole sections; H(z)=H(z,1)H(z,2)...H(z,L/2) where
    L is the number of poles and each section is a ratio of quadratics.

    Parameters
    ----------
    d : ndarray
        Numerator weights of analog filter in 1-pole
        sections. d is dimensioned (L/2 x 2).
    c : ndarray
        Denominator weights, dimensioned same as d.

    Returns
    -------
    b : ndarray
        Digital numerator weights, dimensioned (L/2 x 3).
    a : ndarray
        Digital denominator weights, dimensioned the same.

    """
    L2, ncd = d.shape
    nr, ncc = c.shape

    # Check for errors.
    if(nr != L2 or ncd != 2 or ncc != 2):
        raise Exception('Inputs d and c must both be L/2 x 2 arrays.')

    # Bilinear transformation of H(s) to H(z) using z and p vectors.
    a = np.zeros((L2, 3), dtype=np.double)
    a[:, 0] = np.abs(c[:, 0] + c[:, 1])**2

    if np.min(a[:, 0]) == 0:
        raise Exception('"c" should not have a row of zeros.')
    a[:, 1] = 2*np.real((c[:, 0] + c[:, 1]) * np.conj(c[:, 1] - c[:, 0]))
    a[:, 2] = np.abs(c[:, 1] - c[:, 0])**2

    b = np.zeros((L2, 3), dtype=np.double)
    b[:, 0] = np.abs(d[:, 0] + d[:, 1])**2
    b[:, 1] = 2*np.real((d[:, 0] + d[:, 1]) * np.conj(d[:, 1] - d[:, 0]))
    b[:, 2] = np.abs(d[:, 1] - d[:, 0])**2

    # Scale H(z) so a(:,1)=1:
    sa = np.kron(np.ones((3, 1)), a[:, 0]).T
    a = a / sa
    b = b / sa
    return b, a


def freqz(sosmat, nsamples=44100, sample_rate=44100, plot=True):
    """Plots Frequency response of sosmat."""
    from pylab import np, plt, fft, fftfreq
    x = np.zeros(nsamples)
    x[int(nsamples/2)] = 0.999
    y, states = sosfilter_double_c(x, sosmat)
    Y = fft(y)
    f = fftfreq(len(x), 1.0/sample_rate)
    if plot:
        plt.grid(True)
        plt.axis([0, sample_rate / 2, -100, 5])
        L = 20*np.log10(np.abs(Y[:int(len(x)/2)]) + 1e-17)
        plt.semilogx(f[:int(len(x)/2)], L, lw=0.5)
        plt.hold(True)
        plt.title(u'freqz sos filter')
        plt.xlabel('Frequency / Hz')
        plt.ylabel(u'Damping /dB(FS)')
        plt.xlim((10, sample_rate/2))
        plt.hold(False)
    return x, y, f, Y
