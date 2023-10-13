from abc import ABC, abstractmethod
from pyfilterbank import butterworth


class SosFilterDesigner(ABC):
    def __init__(self, order:int):
        """Returns a FilterDesigner object for filters of given `order`
        """
        self._order = order

    @abstractmethod
    def bandpass(self, freq1:float, freq2:float):
        """Needs to return a SOS matrix

        Args:
            freq1 (float): Normalized kricical frequency 
                from 0 to 0.5.
            freq2 (float): Normalized kricical frequency 
                from 0 to 0.5

        Returns:
            sosmat (ArrayLike): SOS matrix of shape (order, 6)

        """
        pass

    @abstractmethod
    def lowpass(self, freq1):
        """Needs to return a SOS matrix

        Args:
            freq1 (float): Normalized kricical frequency 
                from 0 to 0.5.
        
        Returns:
            sosmat (ArrayLike): SOS matrix of shape (order/2, 6)

        """
        pass

    @abstractmethod
    def highpass(self, freq1):
        """Needs to return a SOS matrix

        Args:
            freq1 (float): Normalized kricical frequency 
                from 0 to 0.5.
            
        Returns:
            sosmat (ArrayLike): SOS matrix of shape (order/2, 6)

        """
        pass

    @property
    def order(self):
        return self._order
    

class ButterworthDesigner(SosFilterDesigner):
    def bandpass(self, freq1, freq2):
        return butterworth.design_sos('bandpass', self.order, freq1, freq2)
    
    def lowpass(self, freq1):
        return butterworth.design_sos('lowpass', self.order, freq1)
    
    def highpass(self, freq1):
        return butterworth.design_sos('highpass', self.order, freq1)