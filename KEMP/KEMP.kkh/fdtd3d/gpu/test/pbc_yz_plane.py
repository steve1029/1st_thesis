# Author  : Ki-Hwan Kim
# Purpose : Test for Pbc class
# Target  : GPU using PyOpenCL
# Created : 2012-01-30
# Modified: 

import numpy as np
import pyopencl as cl
import sys

from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, IncidentDirect, GetFields


nx, ny, nz = 2, 160, 140
tmax, tgap = 150, 10 

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
fields = Fields(context, device, nx, ny, nz)
Pbc(fields, 'xyz')
Core(fields)

tfunc = lambda tstep: np.sin(0.05 * tstep)
IncidentDirect(fields, 'ex', (0, 20, 0), (-1, 20, -1), tfunc) 
#IncidentDirect(fields, 'ex', (0, 0, 20), (-1, -1, 20), tfunc) 
getf = GetFields(fields, 'ex', (0.5, 0, 0), (0.5, -1, -1))

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(np.zeros((ny, nz), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.xlabel('y')
plt.ylabel('z')
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