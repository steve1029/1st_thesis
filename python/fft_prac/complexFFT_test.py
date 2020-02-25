import os, time, datetime, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

um = 1e-6
nm = 1e-9

Lz = 1000 * um
Nz = 256
dz = Lz/Nz
courant = 1./4

Tsteps = 2048
dt = courant * dz / c
t = np.arange(Tsteps) * dt
srt = 100 * um
end = 200 * um
interval = 0.3 * um

wvl = np.arange(srt, end, interval, dtype=np.float64)
freq = c / wvl[::-1]
freqc = (freq[0] + freq[-1]) / 2
omega = freq * 2 * np.pi
wvlc = (wvl[0] + wvl[-1]) / 2
spread = 0.3
w0 = 2 * np.pi * freqc
ws = spread * w0
ts = 1./ws
tc = 500 * dt

src_t = np.exp((-.5) * (((t - tc)*ws)**2)) * np.cos(w0*(t-tc))
"""
dt = 1
t = np.arange(0, 128, dt)
t0 = 32
spread = 128
src_t = np.exp((-.5) * (((t-t0)**2)/spread**2))
"""

src_f = np.fft.fft(src_t)
f = np.fft.fftfreq(len(t), dt)

for i in range(len(src_f)):
	if src_f.real[i] == 0.: src_f.real[i] = 1e-30

magni = abs(src_f)
phase = np.arctan(src_f.imag/src_f.real)
print(phase)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot(t, src_t)
ax2.plot(f, magni)
ax4.plot(f, phase)

ax1.set_xlabel("time step")
ax2.set_xlabel("frequency")
ax4.set_xlabel("frequency")
ax1.set_ylabel("Amplitude")
ax2.set_ylabel("Magnitude")
ax4.set_ylabel("phase")

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
fig.tight_layout()
fig.savefig("./complexFFT_test.png")
