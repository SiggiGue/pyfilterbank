import unittest
import numpy as np
import pyfilterbank.octbank as fb


def zeros(num_samples=1024, num_chan=1):
    return np.zeros((num_samples, num_chan))

def ones(num_samples=1024, num_chan=1):
    return np.ones((num_samples, num_chan))


class TestFractionalOctaveFilterBank(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self. order = 4
        self.filterfuns = ['cffi', 'py']
        self.ofbs = []
        for fun in self.filterfuns:
            self.ofbs += [fb.FractionalOctaveFilterbank(
                sample_rate=self.sample_rate,
                order=self.order)
            ]

    def test_zeros(self):
        for ofb in self.ofbs:
            signal, states = ofb.filter_mimo_c(zeros())
            self.assertTrue(np.sum(np.sum(signal)) == 0)
            signal, states = ofb.filter(zeros())
            self.assertTrue(np.sum(np.sum(signal)) == 0)

    def test_ones(self):
        for ofb in self.ofbs:
            signal, states = ofb.filter_mimo_c(ones())
            self.assertTrue(np.all(signal != 1))
            signal, states = ofb.filter(ones())
            self.assertTrue(np.all(signal != 1))


class TestFilterbankModuleFuntions(unittest.TestCase):
    def setUp(self):
        self.start = -20
        self.stop = 20
        self.fnorm = 1000.0
        self.nth_oct = 3.0
        self.order = 2
        self.sample_rate = 44100

    def test_frequencies_fractional_octaves(self):
        centerfreqs, edges = fb.frequencies_fractional_octaves(
            self.start, self.stop, self.fnorm, self.nth_oct)
        num_bands = self.stop-self.start+1
        self.assertEqual(centerfreqs[+self.start-1], self.fnorm)
        self.assertEqual(len(centerfreqs), num_bands)
        self.assertEqual(len(edges), num_bands+1)
        self.assertTrue(np.all(np.isfinite(centerfreqs)))
        self.assertTrue(np.all(np.isfinite(edges)))

    def test_design_sosmat_band_passes(self):
        centerfreqs, edges = fb.frequencies_fractional_octaves(
            self.start, self.stop, self.fnorm, self.nth_oct)
        sosmat = fb.design_sosmat_band_passes(
            self.order, edges, self.sample_rate)
        self.assertTrue(np.all(np.isfinite(sosmat)))


if __name__ == '__main__':
    unittest.main()
