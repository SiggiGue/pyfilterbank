from pylab import plt
import melbank

f1, f2 = 1000, 8000
melmat, (melfreq, fftfreq) = melbank.compute_melmat(6, f1, f2, num_fft_bands=4097)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(fftfreq, melmat.T)
ax.grid(True)
ax.set_ylabel('Weight')
ax.set_xlabel('Frequency / Hz')
ax.set_xlim((f1, f2))
ax2 = ax.twiny()
ax2.xaxis.set_ticks_position('top')
ax2.set_xlim((f1, f2))
ax2.xaxis.set_ticks(melbank.mel_to_hertz(melfreq))
ax2.xaxis.set_ticklabels(['{:.0f}'.format(mf) for mf in melfreq])
ax2.set_xlabel('Frequency / mel')
plt.tight_layout()

fig, ax = plt.subplots()
ax.matshow(melmat)
plt.axis('equal')
plt.axis('tight')
plt.title('Mel Matrix')
plt.tight_layout()