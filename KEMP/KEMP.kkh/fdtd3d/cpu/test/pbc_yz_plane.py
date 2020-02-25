# Author  : Ki-Hwan Kim
# Purpose : Test for Pbc class
# Target  : CPU
# Created : 2012-02-13
# Modified: 

import numpy as np
import sys

from kemp.fdtd3d.cpu import QueueTask, Fields, Core, Pbc, IncidentDirect, GetFields


nx, ny, nz = 2, 160, 140
#nx, ny, nz = 160, 140, 2
#nx, ny, nz = 160, 2, 140
tmax, tgap = 150, 10 

# instances 
fields = Fields(QueueTask(), nx, ny, nz, use_cpu_core=0)
Pbc(fields, 'xyz')
Core(fields)

tfunc = lambda tstep: np.sin(0.05 * tstep)
# yz-plane
IncidentDirect(fields, 'ex', (0, 20, 0), (-1, 20, -1), tfunc) 
getf = GetFields(fields, 'ex', (0.5, 0, 0), (0.5, -1, -1))
arr_plot = np.zeros((ny, nz), fields.dtype)

# xy-plane
#IncidentDirect(fields, 'ez', (20, 0, 0), (20, -1, -1), tfunc) 
#getf = GetFields(fields, 'ez', (0, 0, 0.5), (-1, -1, 0.5))
#arr_plot = np.zeros((nx, ny), fields.dtype)

# xz-plane
#IncidentDirect(fields, 'ey', (20, 0, 0), (20, -1, -1), tfunc) 
#getf = GetFields(fields, 'ey', (0, 0.5, 0), (-1, 0.5, -1))
#arr_plot = np.zeros((nx, nz), fields.dtype)

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(arr_plot.T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
#plt.xlabel('y')
#plt.ylabel('z')
plt.colorbar()

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.get_event().wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')