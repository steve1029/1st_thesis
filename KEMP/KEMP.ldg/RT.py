import numpy as np
import h5py as h5
import pylab as pyl

savefile = 'square_cavity_1'
f = h5.File('./save/%s.h5' %savefile,'r')

#--------------------------------------------------------------------------------------------------------
freqs = np.array(f.get('freqs'))
r = np.array(f.get('r'))
t = np.array(f.get('t'))

R = abs(r)**2
T = abs(t)**2

fig = pyl.figure()
pyl.xlabel('freqs(THz)')
pyl.plot(freqs,T, color='k',label="Trs")
pyl.plot(freqs,R, color='r',label="Ref")
pyl.legend()
pyl.show()
