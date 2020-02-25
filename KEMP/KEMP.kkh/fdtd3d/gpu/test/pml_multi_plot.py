# Author  : Ki-Hwan Kim
# Purpose : Test for PML class along three axes
# Target  : GPU using PyOpenCL
# Created : 2012-02-15
# Modified: 

import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.gpu import Fields, Core, Pbc, Pml, IncidentDirect, GetFields


tmax = 1000
npml = 10
tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)

# plot
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')
#plt.ion()
fig = plt.figure(figsize=(14,5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

# gpu device
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
device = gpu_devices[0]

# xy-plane
nx, ny, nz = 180, 160, 2
fields = Fields(context, device, nx, ny, nz)
Pbc(fields, 'z')
Pml(fields, ('+-', '+-', ''), npml)
Core(fields)
IncidentDirect(fields, 'ez', (0.4, 0.3, 0), (0.4, 0.3, -1), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

getf.get_event().wait()
ax1.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax1.set_title('xy-plane')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# yz-plane
nx, ny, nz = 2, 180, 160
fields = Fields(context, device, nx, ny, nz)
Pbc(fields, 'x')
Pml(fields, ('', '+-', '+-'), npml)
Core(fields)
IncidentDirect(fields, 'ex', (0, 0.4, 0.3), (-1, 0.4, 0.3), tfunc) 
getf = GetFields(fields, 'ex', (0.5, 0, 0), (0.5, -1, -1))

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

getf.get_event().wait()
ax2.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax2.set_title('yz-plane')
ax2.set_xlabel('y')
ax2.set_ylabel('z')

# xz-plane
nx, ny, nz = 180, 2, 160
fields = Fields(context, device, nx, ny, nz)
Pbc(fields, 'y')
Pml(fields, ('+-', '', '+-'), npml)
Core(fields)
IncidentDirect(fields, 'ey', (0.4, 0, 0.3), (0.4, -1, 0.3), tfunc) 
getf = GetFields(fields, 'ey', (0, 0.5, 0), (-1, 0.5, -1))

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

getf.get_event().wait()
ax3.imshow(getf.get_fields().T, vmin=-1.1, vmax=1.1)
ax3.set_title('xz-plane')
ax3.set_xlabel('x')
ax3.set_ylabel('z')

plt.savefig('plot.png')
#plt.show()
