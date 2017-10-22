import pytest
import numpy as np
from pyfilterbank import octave


def zeros(num_samples=1024, num_chan=1):
    return np.zeros((num_samples, num_chan))


def ones(num_samples=1024, num_chan=1):
    return np.ones((num_samples, num_chan))


class TestBaseFractionalOctaveModel:
    model = octave.BaseFractionalOctaveModel(1000.0, 1.0)

    @pytest.mark.parametrize("centerfreq,fraction", zip(np.logspace(1, 10), range(1, 25)))
    def test_frequencies(self, centerfreq, fraction):
        self.model.centerfreq = centerfreq
        self.model.fraction = fraction
        assert self.model._centerfreq == centerfreq
        assert self.model._fraction == fraction
        
        self.model.base = 10
        expected_loweredge = centerfreq*10**(-0.3/(2*fraction))
        expected_upperedge = centerfreq*10**(0.3/(2*fraction))
        assert np.isclose(
            self.model.upperedgefreq, 
            expected_upperedge)
        assert np.isclose(
            self.model.loweredgefreq, 
            expected_loweredge)
        assert np.isclose(self.model.bandwidth, expected_upperedge-expected_loweredge)
        
        self.model.base = 2
        expected_loweredge = centerfreq*2**(-1.0/(2*fraction))
        expected_upperedge = centerfreq*2**(1.0/(2*fraction))
        assert np.isclose(
            self.model.upperedgefreq, 
            expected_upperedge)
        assert np.isclose(
            self.model.loweredgefreq, 
            expected_loweredge
        )
        assert np.isclose(self.model.bandwidth, expected_upperedge-expected_loweredge)

    def test_property_assignments(self):
        with pytest.raises(ValueError):
            self.model.base = 1
            self.model.centerfreq = '1000 Hz'
            self.model.fraction = '3rd Octave'


class TestFractionalOctaveModel:
    ofreqs = octave.FractionalOctaveModel(bandnum=0, referencefreq=1000)

    def test_bandnum(self):
        self.ofreqs.bandnum = 1.0
        assert self.ofreqs.bandnum == 1
        with pytest.raises(ValueError):
            self.ofreqs.bandnum = 1.001
        self.ofreqs.centerfreq = self.ofreqs.referencefreq
        assert self.ofreqs.bandnum == 0

    def test_fraction(self):
        self.ofreqs.fraction = 12
        self.ofreqs.bandnum = 40
        assert np.isclose(self.ofreqs.centerfreq, 10000.0)        

        self.ofreqs.bandnum = 10
        self.ofreqs.fraction = 3
        assert np.isclose(self.ofreqs.centerfreq, 10000.0)

        self.ofreqs.bandnum = -10
        self.ofreqs.fraction = 3
        assert np.isclose(self.ofreqs.centerfreq, 100.0)

        self.ofreqs.fraction = 6
        self.ofreqs.bandnum = -20   
        assert np.isclose(self.ofreqs.centerfreq, 100.0)

    def test_referencefreq(self):
        self.ofreqs.referencefreq = 100
        self.ofreqs.fraction = 3
        self.ofreqs.bandnum = 10
        assert np.isclose(self.ofreqs.centerfreq, 1000.0)
        self.ofreqs.referencefreq = 1000  # must be set back for other tests.

    @pytest.mark.parametrize("bandnum", range(-40, 50, 10))
    def test_getitem_thirdoct_base10(self, bandnum):
        self.ofreqs.fraction = 3
        self.ofreqs.base = 10
        obf = self.ofreqs[bandnum]
        assert np.isclose(obf.centerfreq, 1000.0*10**(0.1*obf.bandnum))
        
    @pytest.mark.parametrize("bandnum", range(-10, 11))
    def test_getitem_oct_base2(self, bandnum):
        self.ofreqs.fraction = 1
        self.ofreqs.base = 2
        obf = self.ofreqs[bandnum]
        assert np.isclose(obf.centerfreq, 1000.0*2**(obf.bandnum))

    def test_getitem_range(self):
        self.ofreqs.fraction = 3
        self.ofreqs.base = 10
        for obf in self.ofreqs[-30:40:10]:
            assert np.isclose(obf.centerfreq, 1000.0*10**(0.1*obf.bandnum))

    def test_range_by_bandnum_relative(self):
        self.ofreqs.fraction = 3
        self.ofreqs.base = 10
        for obf in self.ofreqs.range_by_bandnum(10, 40, 10, relative=True):
            assert np.isclose(obf.centerfreq, 1000.0*10**(0.1*obf.bandnum))

    @pytest.mark.parametrize("bandoffset", range(-30, 31))
    def test_shift_by(self, bandoffset):
        a = self.ofreqs.shift_by(bandoffset)
        assert(a.bandnum == self.ofreqs.bandnum + bandoffset)

    def test_find_nominal_centerfreq(self):
        self.ofreqs.find_nominal_centerfreq()
        with pytest.raises(Exception):
            self.ofreqs.fraction = 12
            self.ofreqs.bandnum = 9
            self.ofreqs.find_nominal_centerfreq()


def _chkprop(self, prop, val, ckp=True):
    origin_value = getattr(self.o, prop)
    if val == origin_value:
        params = self.o._params()
        sos = self.o.sos
        setattr(self.o, prop, val)      
        if sos.shape == self.o.sos.shape:
            assert np.allclose(self.o.sos, sos)
        if ckp:
            assert self.o._params() == params                      
    else:
        params = self.o._params()
        sos = self.o.sos
        setattr(self.o, prop, val)
        if ckp:
            assert self.o._params() != params
        if sos.shape == self.o.sos.shape:
            assert not np.allclose(self.o.sos, sos)
        setattr(self.o, prop, origin_value)
        if sos.shape == self.o.sos.shape:
            assert np.allclose(self.o.sos, sos)
        if ckp:
            assert self.o._params() == params
        

class TestBaseFractionalOctaveBandFilter(TestFractionalOctaveModel):
    o = octave.BaseFractionalOctaveBandFilter(
        order=4,
        centerfreq=1000,
        fraction=1.0,
        samplerate=44100.0,
        base=10
    )
    _chkprop = _chkprop

    def test_samplerate(self):
        self._chkprop('samplerate', self.o.centerfreq*3)

    @pytest.mark.parametrize("order", range(2, 30,2))
    def test_order(self, order):
        self._chkprop('order', order)
        with pytest.raises(ValueError):
            self._chkprop('order', order+1)

    @pytest.mark.parametrize("centerfreq", np.logspace(0, 4))
    def test_centerfreq(self, centerfreq):
        self._chkprop('centerfreq', centerfreq)

    @pytest.mark.parametrize("fraction", [1, 3, 6, 12, 24])
    def test_fraction(self, fraction):
        self._chkprop('fraction', fraction)

    @pytest.mark.parametrize("base", [2, 10])
    def test_base(self, base):
        self._chkprop('base', base)

    def test_design_sos_func(self):

        def func(order, lef, uef):

            return octave.chebyshev.design_sos(
                'bandpass', order, lef, uef, stopband_gain_db=-80)

        self._chkprop("design_sos_func", func, ckp=False)


class TestFractionalOctaveBandFilter(TestBaseFractionalOctaveBandFilter):
    o = octave.FractionalOctaveBandFilter(
        order=4,
        bandnum=0,
        fraction=1.0,
        samplerate=44100.0,
        base=10
    )
    _chkprop = _chkprop

    @pytest.mark.parametrize("bandnum", range(-4, 4))
    def test_bandnum(self, bandnum):
        self._chkprop('bandnum', bandnum)

    @pytest.mark.parametrize("referencefreq", [10, 100, 1000, 10000])
    def test_referencefreq(self, referencefreq):
        self.o.bandnum = 0
        self._chkprop('referencefreq', referencefreq)
        self.o.referencefreq = referencefreq
        assert self.o.centerfreq == referencefreq
        self.o.referencefreq = octave.DEFAULT_REFERENCEFREQ

    @pytest.mark.parametrize("centerfreq", [10, 100, 1000, 10000])
    def test_centerfreq(self, centerfreq):
        self._chkprop('centerfreq', centerfreq)

    @pytest.mark.parametrize("base", [2, 10])
    def test_base(self, base):
        self._chkprop('base', base)

    def test_design_sos_func(self):

        def func(order, lef, uef):
            
            return octave.chebyshev.design_sos(
                'bandpass', order, lef, uef, stopband_gain_db=-80)

        self._chkprop("design_sos_func", func, ckp=False)
        



# class TestFractionalOctaveFilterBank:
#     samplerate = 44100
#     order = 4
#     filterfuns = ['cffi', 'py']
#     ofbs = []
#     for fun in filterfuns:
#         ofbs += [fb.FractionalOctaveFilterbank(
#             samplerate=samplerate,
#             order=order)
#         ]

#     def test_zeros(self):
#         for ofb in self.ofbs:
#             signal, states = ofb.filter_mimo_c(zeros())
#             self.assertTrue(np.sum(np.sum(signal)) == 0)
#             signal, states = ofb.filter(zeros())
#             self.assertTrue(np.sum(np.sum(signal)) == 0)

#     def test_ones(self):
#         for ofb in self.ofbs:
#             signal, states = ofb.filter_mimo_c(ones())
#             self.assertTrue(np.all(signal != 1))
#             signal, states = ofb.filter(ones())
#             self.assertTrue(np.all(signal != 1))


# class TestFilterbankModuleFuntions:

#     start = -20
#     stop = 20
#     fnorm = 1000.0
#     nth_oct = 3.0
#     order = 2
#     samplerate = 44100

#     def test_frequencies_fractional_octaves(self):
#         centerfreqs, edges = fb.frequencies_fractional_octaves(
#             self.start, self.stop, self.fnorm, self.nth_oct)
#         num_bands = self.stop-self.start+1
#         self.assertEqual(centerfreqs[+self.start-1], self.fnorm)
#         self.assertEqual(len(centerfreqs), num_bands)
#         self.assertEqual(len(edges), num_bands+1)
#         self.assertTrue(np.all(np.isfinite(centerfreqs)))
#         self.assertTrue(np.all(np.isfinite(edges)))

#     def test_design_sosmat_band_passes(self):
#         centerfreqs, edges = fb.frequencies_fractional_octaves(
#             self.start, self.stop, self.fnorm, self.nth_oct)
#         sosmat = fb.design_sosmat_band_passes(
#             self.order, edges, self.samplerate)
#         self.assertTrue(np.all(np.isfinite(sosmat)))

if __name__ == '__main__':
    import pytest
    pytest.main()
