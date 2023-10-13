from collections.abc  import Iterable
from itertools import cycle
from scipy.signal import sosfilt
from pyfilterbank.scales import Band, FractionalOctaveScale
from pyfilterbank.design import SosFilterDesigner, ButterworthDesigner


class Filterbank:
    def __init__(self,
                 samplerate,
                 bands:Iterable[Band], 
                 designer:SosFilterDesigner,
                 lp:bool=False,
                 hp:bool=False):
        self._samplerate = samplerate
        self._bands = tuple(b for b in bands if b.upper<=samplerate/2)
        self._designer = designer
        self._lp = lp  # TODO
        self._hp = hp  # TODO
        self._filters = tuple(self._design_filters())

    def _design_filters(self):

        if self._lp and hasattr(self._designer, 'lowpass'):
            yield self._designer.lowpass(min(self._bands).lower/self._samplerate) 

        for band in self._bands:
            yield self._designer.bandpass(
                freq1=band.lower/self._samplerate, 
                freq2=band.upper/self._samplerate)
        
        if self._hp and hasattr(self._designer, 'highpass'):
            yield self._designer.highpass(max(self._bands).upper/self._samplerate)

    def filter(self, sig, axis=-1, states=None):
        if states is None:
             states = cycle(states)
        for sos, state in zip(self._filters, states):
            yield sosfilt(sos, sig, axis=axis, zi=state)


if __name__ == "__main__":
    import scipy.signal
    import matplotlib.pyplot as plt
    fb = Filterbank(44100, FractionalOctaveScale(fraction=3).range_by_freq(20, 20000), ButterworthDesigner(order=6), lp=True, hp=True)
    for sos in fb._filters: 
        w,h = scipy.signal.sosfreqz(sos, 2*8192)
        plt.semilogx(w, 20*np.log10(np.abs(h)))