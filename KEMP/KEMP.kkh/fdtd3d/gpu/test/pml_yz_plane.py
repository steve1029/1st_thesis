# Author  : Ki-Hwan Kim
# Purpose : Test for Pml class (yz-plane)
# Target  : GPU using PyOpenCL
# Created : 2012-01-30
# Modified: 

import numpy as np
import pyopencl as cl
import sys

from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, Pml, IncidentDirect, GetFields


nx, ny, nz = 2, 250, 300
tmax, tgap = 1000, 10 
npml = 10

# instances
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
fields = Fields(context, device, nx, ny, nz)
Pbc(fields, 'x')
Pml(fields, ('', '+-', '+-'), npml)
Core(fields)

tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)
IncidentDirect(fields, 'ex', (0, 0.4, 0.3), (-1, 0.4, 0.3), tfunc) 
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

from matplotlib.patches import Rectangle
rects = [Rectangle((ny-npml, 0), npml, nz, alpha=0.1), \
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

        getf.get_event().wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
