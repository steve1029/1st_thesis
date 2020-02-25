# Author  : Ki-Hwan Kim
# Purpose : Test for Drude class (yz-plane)
# Target  : GPU using PyOpenCL
# Created : 2012-01-30
# Modified: 

import numpy as np
import pyopencl as cl
import sys

from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, Pml, Drude, IncidentDirect, GetFields


nx, ny, nz = 2, 500, 600
tmax, tgap = 1000, 10 
npml = 10

# instances
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]
fields = Fields(context, device, nx, ny, nz)

length = 1e-8       # 10 nm
c0 = 299792458      # m/s
ep_inf = 9.0685
drude_freq = 2 * np.pi * 2155.6 * 1e12 * length / c0
gamma = 2 * np.pi * 18.36 * 1e12 * length / c0
Drude(fields, (0, 0.6, 0), (-1, -1, -1), ep_inf, drude_freq, gamma)

Pbc(fields, 'x')
Pml(fields, ('', '-', '+-'), npml)
Core(fields)

wavelength = 49.5   # 495 nm 
tfunc = lambda tstep: 50 * np.sin(2 * np.pi / wavelength * tstep)
IncidentDirect(fields, 'ex', (0, 0.4, 0.5), (-1, 0.4, 0.5), tfunc) 

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
