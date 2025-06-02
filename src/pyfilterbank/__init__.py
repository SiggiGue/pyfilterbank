"""The package :mod:`pyfilterbank` provides tools for acousticians and audiologists/engineers working with python.

A fractional octave filter bank is provided in the module :mod:`octbank`. You can use it to split your signals into many bands of constant relative fractional octave band width. The output signals stay in the same domain as the input signal but are band passed groups of it. The filtering routines are placed in :mod:`sosfiltering` and the filter design functionality is implemented in :mod:`butterworth`.

Spectral weigthing for level measurements can be done with the tools in :mod:`splweighting`.
For fft-based and more physiological motivated filtering there is the module :mod:`melbank` with some tools for transforming linear spectra to mel-spectra.
A gammatone filter bank is planned but not implemented yet.

"""

from . import butterworth
from . import melbank
from . import octbank
from . import rbj_audio_eq
from . import sosfiltering
from . import splweighting
from . import stft
from . import gammatone

#from .sosfiltering import sosfilter
from .octbank import FractionalOctaveFilterbank
from .gammatone import GammatoneFilterbank

__version__ = '0.0.0'
__all__ = ['butterworth', 'melbank', 'octbank', 'rbj_audio_eq', 'sosfiltering', 'splweighting', 'stft']
