"""The package :mod:`pyfilterbank` provides tools for acousticians and audiologists/engineers working with python.

A fractional octave filter bank is provided in the module :mod:`octbank`. You can use it to split your signals into many bands of constant relative fractional octave band width. The output signals stay in the same domain as the input signal but are band passed groups of it. The filtering routines are placed in :mod:`sosfiltering` and the filter design functionality is implemented in :mod:`butterworth`.

Spectral weighting for level measurements can be done with the tools in :mod:`splweighting`.
For fft-based and more physiological motivated filtering there is the module :mod:`melbank` with some tools for transforming linear spectra to mel-spectra.
A gammatone filter bank is planned but not implemented yet.

"""
