import numpy as np
import sys

from kemp.fdtd3d.naive import Fields, Core, Pbc, IncidentDirect, Pml


nx, ny, nz = 2, 250, 300
tmax, tgap = 1000, 10 
npml = 10

# instances 
fields = Fields(nx, ny, nz)
Pbc(fields, 'x')
Pml(fields, ('', '+-', '+-'), npml) #, kappa_max=1.01
Core(fields)

tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)
#IncidentDirect(fields, 'ex', (0, -50, 0.5), (-1, -50, 0.5), tfunc) 
#IncidentDirect(fields, 'ex', (0, 50, 0.5), (-1, 50, 0.5), tfunc) 
#IncidentDirect(fields, 'ex', (0, 0.5, -50), (-1, 0.5, -50), tfunc) 
IncidentDirect(fields, 'ex', (0, 0.5, 50), (-1, 0.5, 50), tfunc) 

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
imag = plt.imshow(np.zeros((ny, nz), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.xlabel('y')
plt.xlabel('z')
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

        imag.set_array( fields.ex[nx/2,:,:].T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
