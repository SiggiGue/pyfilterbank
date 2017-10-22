import numpy as np
import sys
sys.path.append("..")
import pyfilterbank.sosfiltering as sf
import pyfilterbank.butterworth as bw


class TestButterSos:
    samplerate = 44100
    order =  2
    v1 = 0.1272
    v2 = 0.3541

    def test_lowpass(self):
        sosmat = bw.design_sos('lowpass', 2, self.v1)
        assert(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, samplerate=self.samplerate, plot=False)
        mask = F >= self.v1*self.samplerate
        assert(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))

    def test_highpass(self):
        sosmat = bw.design_sos('highpass', 2, self.v1)
        assert(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, samplerate=self.samplerate, plot=False)
        mask = np.logical_and(F <= self.v1*self.samplerate, F >=0)
        assert(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))

    def test_bandpass(self):
        sosmat = bw.design_sos('bandpass', 2, self.v1, self.v2)
        assert(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, samplerate=self.samplerate, plot=False)
        mask = np.logical_and(F <= self.v1*self.samplerate,
                                    F >= self.v2*self.samplerate)
        assert(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))

    def test_bandstop(self):
        sosmat = bw.design_sos('bandstop', 2, self.v1, self.v2)
        assert(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, samplerate=self.samplerate, plot=False)
        mask = np.logical_and(F >= self.v1*self.samplerate,
                                    F <= self.v2*self.samplerate)
        assert(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))


if __name__ == '__main__':
    import pytest
    
    pytest.main()
