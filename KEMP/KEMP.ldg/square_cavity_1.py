import numpy as np
import h5py as h5
from KEMP import Basic_FDTD, Dielectric, structures, to_epr, to_SI, to_NU
from datetime import datetime as dtm
import sys
import KEMP.geometry as geo 
from KEMP.materials import Au

c = 299792458. 
nm = 1e-9
um = 1e-6
mm = 1e-3
THz = 1e12
freq0 = c/(600.*um)

#wavelength = np.arange(400,800,5)*nm
freqs = np.arange(0.3, 2.0, 0.01)*THz
src_pos= 300
trs_pos= 2700

dx, dy, dz = [1*um, 1*um, 1*um]
nx, ny, nz = [50, 50, 3000]

x = np.ones(nx, dtype=np.float64)*dx 
y = np.ones(ny, dtype=np.float64)*dy 
z = np.ones(nz, dtype=np.float64)*dz

space_grid = (x, y ,z)
lx, ly, lz = dx*(nx+1), dy*(ny+1), dz*(nz+1)

fdtd = Basic_FDTD('3D', space_grid, dtype=np.complex64, engine='intel_cpu')

pml_apply = {'x':'' , 'y':''  , 'z':'+-'  }
pbc_apply = {'x':True, 'y':True, 'z':False}

fdtd.apply_PML(pml_apply)
fdtd.apply_PBC(pbc_apply)

wfreq0_NU = to_NU(fdtd, 'angular frequency', 2.*np.pi*freq0)

#----------------------------------structure-------------------------------------------------------------------

xc = fdtd.x[fdtd.nx/2]
yc = fdtd.y[fdtd.ny/2]
zc = fdtd.z[fdtd.nz/2]

pec = Dielectric(to_epr(fdtd, n=10.**10))
air = Dielectric(to_epr(fdtd, n=1.))
GaAs = Dielectric(to_epr(fdtd,n=3.5))

a1 = 36*um
a2 = 28*um
b1 = 36*um
b2 = 28*um
c1 = 10*um
c2 = 4 *um
c3 = 2 *um
d1 = 4 *um
thickness = 200 * nm

slab = structures.Box(GaAs, ((0.*um, 0.*um, 1500*um), (lx, ly, 1600*um)))
mat  = geo.square_cavity_1(Au,air,(xc,yc,zc),a1,a2,b1,b2,c1,c2,c3,d1,thickness,angle=0.,rot_axis=(0.,0.,1.))

fdtd.set_structures([slab])

#--------------------------------------------------------------------------------------------------------------

inc    = fdtd.apply_direct_source('ey', ((0,0,src_pos),(-1,-1,src_pos)))


import h5py as h5
f = h5.File('./save/%s.h5' %'sttest','w')
f.create_dataset('ep', data=0.5/fdtd.ce2x[:,:,nz/2-50:nz/2+50])


print 'Setting Complete and READY to RUN'

dt_SI = to_SI(fdtd, 'time', fdtd.dt)
#ts = min(wavelength)/c
ts = 1. / max(freqs)
tc = 5. * ts/dt_SI
t0 = dtm.now()

tmax = 30000

ey_src = np.zeros(tmax, dtype=np.complex64)
ey_trs = np.zeros(tmax, dtype=np.complex64)

for tstep in xrange(tmax):

    src = np.exp(-((tstep-tc)*dt_SI)**2/(ts**2))*np.cos(wfreq0_NU*tstep*dt_SI)

    ey_src[tstep] = fdtd.ey[:,:,src_pos].mean()
    ey_trs[tstep] = fdtd.ey[:,:,trs_pos].mean()
    
    inc.set_source(src)

    if tstep % 2000 == 0:
        print tstep, (tstep*100/tmax),'%',dtm.now() - t0

    fdtd.updateE()
    fdtd.updateH()

print('FDTD Simulation Complete')

dt_SI = to_SI(fdtd, 'time', fdtd.dt)
tsteps = np.arange(tmax, dtype=fdtd.dtype)
t = tsteps*dt_SI
#freqs = c/wavelength
wavelength = c / freqs
src = np.exp(-((tsteps-tc)*dt_SI)**2/(ts**2))*np.cos(wfreq0_NU*tsteps*dt_SI)
d1=1000*nm
d2=1000*nm
src_ft = ((dt_SI/(2.*np.pi))*src[np.newaxis,:]*np.exp(2j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)
ey_src_ft = ((dt_SI/(2.*np.pi))*ey_src[np.newaxis,:]*np.exp(2.j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)
ey_trs_ft = (np.exp(-1j*2*np.pi*(d2+d1)/wavelength[:,np.newaxis])*(dt_SI/(2.*np.pi))*ey_trs[np.newaxis,:]*np.exp(2.j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)

print tstep, dtm.now() - t0

T = np.zeros(len(wavelength),dtype=np.complex64)
R = np.zeros(len(wavelength),dtype=np.complex64)

T = ey_trs_ft/src_ft
R = np.exp(-1j*4*np.pi*d1/wavelength)*(ey_src_ft-src_ft)/src_ft

ep1=1.
mu1=1.
ep2=1.
mu2=1.

f = h5.File('./save/%s.h5' %'square_cavity_1','w')
f.create_dataset('h', data=thickness)
f.create_dataset('r', data=R)
f.create_dataset('t', data=T)
f.create_dataset('ep1', data=ep1)
f.create_dataset('mu1', data=mu1)
f.create_dataset('ep3', data=ep2)
f.create_dataset('mu3', data=mu2)
f.create_dataset('wavelength', data=wavelength)
f.create_dataset('freqs', data=freqs)
f.close()
