# Author  : Ki-Hwan Kim
# Purpose : Test the Drude class
# Target  : CPU using Numpy
# Created : 2012-01-28
# Modified: 

import numpy as np
import sys

from kemp.fdtd3d.naive import Fields, Core, Pbc, IncidentDirect, Pml, Drude


nx, ny, nz = 2, 250, 300
tmax, tgap = 1000, 10 
npml = 10

# instances 
fields = Fields(nx, ny, nz)
Drude(fields, (0, -30, 0), (-1, -1, -1), \
      ep_inf=9.0685, drude_freq=2*np.pi*2155.6*1e12, gamma=2*np.pi*18.36*1e12)
Pbc(fields, 'x')
Pml(fields, ('', '-', '+-'), npml)
Core(fields)

tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)
IncidentDirect(fields, 'ex', (0, 0.6, 0.5), (-1, 0.6, 0.5), tfunc) 

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(np.zeros((ny, nz), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.xlabel('y')
plt.xlabel('z')
plt.colorbar()

from matplotlib.patches import Rectangle
rects = [Rectangle((ny-30, 0), 30, nz, color='r', alpha=0.1), \
         Rectangle((0, 0), npml, nz, alpha=0.1), \
         Rectangle((0, nz-npml), ny, npml, alpha=0.1), \
         Rectangle((0, 0), ny, npml, alpha=0.1)]
for rect in rects:
    plt.gca().add_patch(rect)

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        imag.set_array( fields.ex[nx/2,:,:].T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
