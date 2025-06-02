import unittest
import numpy as np
import sys
sys.path.append("..")
import pyfilterbank.sosfiltering as sf
import pyfilterbank.butterworth as bw


class ButterSosTestCase(unittest.TestCase):

    def setUp(self):
        self.types = 'lowpass', 'highpass', 'bandpass', 'bandstop'
        self.sample_rate = 44100
        self.order =  2
        self.v1 = 0.1272
        self.v2 = 0.3541

    def test_lowpass(self):
        sosmat = bw.butter_sos('lowpass', 2, self.v1)
        self.assertTrue(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, sample_rate=self.sample_rate, plot=False)
        mask = F >= self.v1*self.sample_rate
        self.assertTrue(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))

    def test_highpass(self):
        sosmat = bw.butter_sos('highpass', 2, self.v1)
        self.assertTrue(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, sample_rate=self.sample_rate, plot=False)
        mask = np.logical_and(F <= self.v1*self.sample_rate, F >=0)
        self.assertTrue(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))

    def test_bandpass(self):
        sosmat = bw.butter_sos('bandpass', 2, self.v1, self.v2)
        self.assertTrue(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, sample_rate=self.sample_rate, plot=False)
        mask = np.logical_and(F <= self.v1*self.sample_rate,
                                    F >= self.v2*self.sample_rate)
        self.assertTrue(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))

    def test_bandstop(self):
        sosmat = bw.butter_sos('bandstop', 2, self.v1, self.v2)
        self.assertTrue(np.all(np.isfinite(sosmat)))
        x, y, F, Y = sf.freqz(
            sosmat, sample_rate=self.sample_rate, plot=False)
        mask = np.logical_and(F >= self.v1*self.sample_rate,
                                    F <= self.v2*self.sample_rate)
        self.assertTrue(np.all(np.abs(Y[mask]) <= 1.0/np.sqrt(2)))


if __name__ == '__main__':
    unittest.main()
