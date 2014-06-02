import unittest
import numpy as np
import sys
sys.path.append("..")
import pyfilterbank.sosfiltering as sf


class SosFilterTestCase(unittest.TestCase):

    def setUp(self):
        self.signal = np.random.randn(1000)
        self.frames = len(self.signal)
        b = [0.5, 0.0, 0.5]
        a = [1.0, -0.8, 0.7]
        sos = [b + a]
        sos = np.array(sos + sos).astype(np.float32)
        self.sos = sos
        self.ksos = int(len(sos)/5)

    def test_implementation(self):
        """c-implementation and python implementation should be equal"""
        op, sp = sf.sosfilter_py(self.signal.copy(), self.sos)
        oc, sc = sf.sosfilter_double_c(self.signal.copy(), self.sos)
        opc, spc = sf.sosfilter_cprototype_py(
            self.signal.copy(), self.sos, None)
        sop = np.sum(np.abs(op))
        soc = np.sum(np.abs(oc))
        sopc = np.sum(np.abs(opc))

        self.assertAlmostEqual(sop, soc, places=6)
        self.assertAlmostEqual(sop, sopc, places=6)
        self.assertAlmostEqual(soc, sopc, places=6)

    def test_zeros(self):
        oc, sc = sf.sosfilter_c(np.zeros(100), self.sos)
        self.assertTrue(np.all(oc == np.zeros(100)))

    def test_ones(self):
        oc, sc = sf.sosfilter_c(np.ones(100), self.sos)
        self.assertTrue(np.all(oc != np.ones(100)))


if __name__ == '__main__':
    unittest.main()
