from numpy import sqrt, pi, cos, sin, sinh, log


def rbj_sos(filtertype, sample_rate, f0, gain_db=None,
            q_factor=None, band_width=None, shelf_slope=None):

    if 'shelf' in filtertype and not shelf_slope:
        raise(ValueError('shelf_slope mus be specified.'))

    w0 = 2*pi * f0/sample_rate
    amplitude = None if not gain_db else sqrt(10**(gain_db/20.0))
    alpha = _compute_alpha(amplitude, w0, q_factor, band_width, shelf_slope)
    params = {'amplitude': amplitude, 'w0': w0, 'alpha': alpha}

    filterfun = _filtertype_to_filterfun_dict[filtertype]
    sos = filterfun(**params)
    return sos


class RbjEqCascade:
    def __init__(self, sample_rate):
        self._sample_rate = sample_rate
        self._sosmat = []
        self._filterlist = []

    def add(self, filtertype):
        self._filtertypelist += [filtertype]
        filtobj = RbjEq(filtertype, self._sample_rate)
        self._filterlist += [filtobj]
        self._sosmat += [filtobj.sos]



class RbjEq:
    def __init__(self, filtertype, sample_rate, params=None):
        self._filtertype = filtertype
        self._sample_rate = sample_rate
        self._filterfun = _filtertype_to_filterfun_dict[filtertype]
        if not params:
            params, param_names = _get_params_filtertype(filtertype)
        self._params = params
        self._update(**params)

    def update(self, f0,
               gain_db=None,
               q_factor=None,
               band_width=None,
               shelf_slope=None):
        w0 = 2*pi * f0/self.sample_rate
        amplitude = None if not gain_db else sqrt(10**(gain_db/20.0))
        alpha = _compute_alpha(amplitude, w0, q_factor, band_width, shelf_slope)
        params = {'amplitude': amplitude, 'w0': w0, 'alpha': alpha}
        self._sos = self._filterfun(**params)

    @property
    def sos(self):
        return self._sos
    @property
    def params(self):
        return self._params
    @params.setter
    def params(self, value):
        self._params = value
        self.update(**self.params)


def _compute_alpha(amplitude=None, w0=None, q_factor=None,
                   band_width=None,
                   shelf_slope=None):
    if q_factor:
        return sin(w0) / (2*q_factor)
    elif band_width:
        return sin(w0) * sinh(0.5*log(2.0) * band_width * w0/sin(w0))
    elif shelf_slope:
        return sin(w0) / 2 * sqrt((amplitude + 1/alpha) * (1/shelf_slope - 1) +2)
    else:
        raise(ValueError(
            '''You need to specify at least one of:
            q_factor, band_width or shelf_slope.'''))

def _lowpass(w0, alpha):
    b0 = (1 - cos(w0)) / 2.0
    b1 =  1 - cos(w0)
    b2 = (1 - cos(w0)) / 2.0
    a0 =  1 + alpha
    a1 = -2 * cos(w0)
    a2 =  1 - alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _highpass(w0, alpha):
    b0 =  (1 + cos(w0)) / 2.0
    b1 = -(1 + cos(w0))
    b2 =  (1 + cos(w0)) / 2.0
    a0 =   1 + alpha
    a1 =  -2 * cos(w0)
    a2 =   1 - alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _bandpassQ(w0, alpha):
    b0 =  sin(w0) / 2.0  # = Q*alpha
    b1 =  0.0
    b2 = -sin(w0) / 2.0  # = -Q*alpha
    a0 =  1 + alpha
    a1 = -2 * cos(w0)
    a2 =  1 - alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _bandpass(w0, alpha):
    b0 =  alpha
    b1 =  0.0
    b2 = -alpha
    a0 =  1 + alpha
    a1 = -2 * cos(w0)
    a2 =  1 - alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _notch(w0, alpha):
    b0 =  1.0
    b1 = -2 * cos(w0)
    b2 =  1.0
    a0 =  1 + alpha
    a1 = -2 * cos(w0)
    a2 =  1 - alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _apf(w0, alpha):
    b0 =  1 - alpha
    b1 = -2 * cos(w0)
    b2 =  1 + alpha
    a0 =  1 + alpha
    a1 = -2 *cos(w0)
    a2 =  1 - alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _peq(amplitude, w0, alpha):
    b0 =  1 + alpha*amplitude
    b1 = -2 * cos(w0)
    b2 =  1 - alpha*amplitude
    a0 =  1 + alpha/amplitude
    a1 = -2 * cos(w0)
    a2 =  1 - alpha/amplitude
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _lowshelf(amplitude, w0, alpha):
    b0 =    amplitude*((amplitude+1) - (amplitude-1)*cos(w0) + 2*sqrt(amplitude)*alpha)
    b1 =  2*amplitude*((amplitude-1) - (amplitude+1)*cos(w0))
    b2 =    amplitude*((amplitude+1) - (amplitude-1)*cos(w0) - 2*sqrt(amplitude)*alpha)
    a0 =               (amplitude+1) + (amplitude-1)*cos(w0) + 2*sqrt(amplitude)*alpha
    a1 =           -2*((amplitude-1) + (amplitude+1)*cos(w0))
    a2 =               (amplitude+1) + (amplitude-1)*cos(w0) - 2*sqrt(amplitude)*alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

def _highshelf(amplitude, w0, alpha):
    b0 =    amplitude*((amplitude+1) + (amplitude-1)*cos(w0) + 2*sqrt(amplitude)*alpha)
    b1 = -2*amplitude*((amplitude-1) + (amplitude+1)*cos(w0))
    b2 =    amplitude*((amplitude+1) + (amplitude-1)*cos(w0) - 2*sqrt(amplitude)*alpha)
    a0 =               (amplitude+1) - (amplitude-1)*cos(w0) + 2*sqrt(amplitude)*alpha
    a1 =            2*((amplitude-1) - (amplitude+1)*cos(w0))
    a2 =               (amplitude+1) - (amplitude-1)*cos(w0) - 2*sqrt(amplitude)*alpha
    sos = array([b0, b1, b2, a0, a1, a2]) / a0
    return sos

_filtertype_to_filterfun_dict = {
    'lowpass': _lowpass,
    'highpass': _highpass,
    'bandpassQ': _bandpassQ,
    'bandpass': _bandpass,
    'notch': _notch,
    'apf': _apf,
    'peq': _peq,
    'lowshelf': _lowshelf,
    'highshelf': _highshelf,
}
available_filtertypes = list(_filtertype_to_filterfun_dict.keys())
