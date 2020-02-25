import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

nm = 1.e-9
dx = .1*nm
dt = .5*dx/c

nt = 200000
ts = np.arange(nt, dtype=np.float64)*dt - (nt*.5*dt)

nf = 1000
nw = 1000
#wave_min = 10*nm
wave_min = 400*nm
wave_mid = 500*nm
wave_max = 600*nm
freq_min = c/wave_max
freq_mid = c/wave_mid
freq_max = c/wave_min
fs = np.arange(freq_min, freq_max, (freq_max-freq_min)/nf, dtype=np.float64)
ws = c/fs
delta_w = 50*nm
delta_f = freq_mid - (c/(600*nm))

#f_dist = np.exp(-((ws-wave_mid)/delta_w)**2)
#f_dist = np.exp(-((fs-freq_mid)/delta_f)**2)

f_dist = np.ones_like(fs, dtype=np.float64)

#plt.plot(fs, f_dist.real, 'b')
#plt.plot(c/ws, w_dist.real, 'b')
plt.plot(fs, f_dist.real, 'b')
plt.show()

def DFT(t_dist, freqs, times):
    nax = np.newaxis
    dt = times[1] - times[0]
    f_dist_t = (dt/(2.*np.pi))*t_dist[nax,:]*np.exp(-2.j*np.pi*freqs[:,nax]*times[nax,:])
    return f_dist_t.sum(1)

def iDFT(f_dist, freqs, times):
    nax = np.newaxis
    df = freqs[1] - freqs[0]
    t_dist_f = df*f_dist[nax,:]*np.exp(-2.j*np.pi*freqs[nax,:]*times[:,nax])
    return t_dist_f.sum(1)

t_dist = iDFT(f_dist, fs, ts)

plt.clf()
plt.plot(ts, t_dist.real, 'b')
plt.plot(ts, t_dist.imag, 'r')
plt.show()


