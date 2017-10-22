__test__ = False

import imp
import numpy as np
import pyfilterbank
from pysoundfile import SoundFile
from pysoundcard import Stream


filename = r'bonsai.wav'

ofb = filterbank.OctaveFilterbank(nth_oct=3.0, start_band=-18, end_band=12, lphp_bounds=True, filterfun='cffi')

st = Stream(input_device=False)
sf = SoundFile(filename)

st.start()

def play_further(nblocks=120, blen=4096):
    states = np.zeros(2)
    for i in range(nblocks):
        x = sf.read(blen)[:,0]
        y, states = ofb.filter(x,states=states)
        out = y.sum(axis=1).values.flatten()
        st.write(out)

play_further()
