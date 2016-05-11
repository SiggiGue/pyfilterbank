import pyfilterbank as pfb
from pylab import *


fo = pfb.FractionalOctaveFilterbank()
x = randn(10*44100)
y, s = fo.filter(x)
ymr, smr = fo.mrfilter(x)


ly = [10*log10(mean(b*b)) for b in list(y.T)]
lymr = [10*log10(mean(b*b)) for b in ymr]

plot(ly)
plot(lymr)

ymrsb = [fo.mrfilter(xx)[0] for xx in list(y.T)]
ysb = [fo.filter(xx)[0] for xx in list(y.T)]
for sb, sbmr in zip(ysb, ymrsb):
    lsb = [10*log10(mean(b*b)) for b in list(sb.T)]
    lsbmr = [10*log10(mean(b*b)) for b in sbmr]
    figure()
    plot(lsb)
    plot(lsbmr)
