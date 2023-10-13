from abc import ABC, abstractmethod
from typing import Self
import numpy as np
from dataclasses import dataclass


class Scale(ABC):
    """A Frequency band scale.

    """
    
    @abstractmethod
    def freq_to_bandnum(self, freq:float):
        pass

    @abstractmethod
    def bandnum_to_freq(self, bandnum:int):
        pass


    @abstractmethod
    def band_at_bandnum(self, bandnum:int):
        pass

    def band_at_freq(self, freq:float):
        return self.band_at_bandnum(self.freq_to_bandnum(freq))

    def range_by_freq(self, lowest:float, uppermost:float):
        yield from self.range_by_bandnum(
            lowest=self.freq_to_bandnum(lowest),
            uppermost=self.freq_to_bandnum(uppermost)
            )

    def range_by_bandnum(self, lowest:int, uppermost:int, step:int=1):
        for num in range(lowest, uppermost+1, step):
            yield self.band_at_bandnum(num)

    def __getitem__(self, items):
        if isinstance(items, int):
            return self.band_at_bandnum(items)
        elif isinstance(items, slice):
            return self.range_by_bandnum(items.start, items.stop, items.step or 1)


@dataclass(frozen=True)
class Band:
    num: int
    center: float
    lower: float
    upper: float
    width: float
    scale: Scale

    def step_up(self, step:int=1):
        return self.scale.band_at_bandnum(self.num+step)

    def step_down(self, step:int=1):
        return self.scale.band_at_bandnum(self.num-step)
    
    def __lt__(self, other:Self):
        return self.num < other.num

class FractionalOctaveScale(Scale):
    
    def __init__(self, fraction, reference_freq=1000, base=10):
        self._fraction = float(fraction)
        self._referencefreq = float(reference_freq)
        if (base != 10) and (base != 2):
            raise ValueError('`base` must be 10 or 2.')
        self._base = base
        self._halfbw_factor = self._get_halfbw_factor()

    def _get_halfbw_factor(self):
        """Returns the half bandwidth factor."""
        
        if self._base == 10:
            return 10**(0.3/(2*self._fraction))
        else:
            return 2**(1.0/(2*self._fraction))

    def freq_to_bandnum(self, freq):
        """Returns bandnum from given frequency."""
        if self._base == 2:
            bandnum = self._fraction * np.log2(freq/self._referencefreq)
        elif self._base == 10:
            bandnum = (self._fraction / 0.3) * np.log10(freq/self._referencefreq)
        return int(np.round(bandnum))
    
    def bandnum_to_freq(self, bandnum):
        """Returns frequency from given bandnum."""
        if self._base == 2:
            return self._referencefreq * 2.0**(bandnum/self._fraction)
        elif self._base == 10:
            return self._referencefreq * 10.0**(0.3 * bandnum/self._fraction)
            
    def band_at_bandnum(self, num:int) -> Band:
        center = self.bandnum_to_freq(num)
        hbwf = self._halfbw_factor
        lower = center / hbwf
        upper = center * hbwf
        width = (upper - lower)
        return Band(
            num=num,
            center=center,
            lower=lower,
            upper=upper,
            width=width,
            scale=self
            )
    
    # def nearest_nominal_frequency(self, freq):

