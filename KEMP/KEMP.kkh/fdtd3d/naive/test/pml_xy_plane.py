import numpy as np

import sys, os
#sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.naive import Fields, Core, Pbc, IncidentDirect, Pml

nx, ny, nz = 250, 300, 4
tmax, tgap = 3000, 50
npml = 10

# instances 
fields = Fields(nx, ny, nz)
cex, cey, cez = fields.get_ces()
cex[:,:,:] /= 4.
cey[:,:,:] /= 4.
cez[:,:,:] /= 4.
Core(fields)
Pbc(fields, 'z')
Pml(fields, ('+-', '+-', ''), npml)

tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)
IncidentDirect(fields, 'ez', (-50, 0.5, 0), (-50, 0.5, -1), tfunc) 
#IncidentDirect(fields, 'ez', (50, 0.5, 0), (50, 0.5, -1), tfunc) 
#IncidentDirect(fields, 'ez', (0.5, -50, 0), (0.5, -50, -1), tfunc) 
#IncidentDirect(fields, 'ez', (0.3, 0.3, 0), (0.3, 0.3, -1), tfunc) 

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.xlabel('x')
plt.xlabel('y')
plt.colorbar()

from matplotlib.patches import Rectangle
rects = [Rectangle((nx-npml, 0), npml, ny, alpha=0.1), \
         Rectangle((0, 0), npml, ny, alpha=0.1), \
         Rectangle((0, ny-npml), nx, npml, alpha=0.1), \
         Rectangle((0, 0), nx, npml, alpha=0.1)]
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

        imag.set_array( fields.ez[:,:,nz/2].T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
