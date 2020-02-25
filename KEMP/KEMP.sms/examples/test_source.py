import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import sys
from KEMP import Basic_FDTD, Dielectric, Dimagnetic, Dielectromagnetic, to_epr, to_SI, to_NU, Gold, Silver, PA
from KEMP import structures as stc

nm = 1e-9
dx, dy, dz = [10*nm, 10*nm, 10*nm]
nx, ny, nz = [2, 2, 2000] 
x = np.ones(nx, dtype=np.float64)*dx # 0 ~ 20000 nm
y = np.ones(ny, dtype=np.float64)*dy # 0 ~ 10 nm
z = np.ones(nz, dtype=np.float64)*dz # 0 ~ 10 nm
lx, ly, lz = dx*nx, dy*ny, dz*nz
space_grid = (x, y ,z)
engine_name = 'nvidia_cuda'
fdtd = Basic_FDTD('3D', space_grid, dtype=np.complex64, engine=engine_name, device_id=0)

pml_apply = {'x':'' , 'y':''  , 'z':'+-'  }
pbc_apply = {'x':True, 'y':True, 'z':False}
fdtd.apply_PML(pml_apply)
fdtd.apply_PBC(pbc_apply)

#mat = Dielectric(epr=4.)
#mat = Dimagnetic(mur=9.)
mat = Dielectromagnetic(epr=4., mur=9.)
slab = stc.Box(mat, ((-1.*nm,-1.*nm,-1.*nm),(lx+1.*nm,ly+1.*nm,lz+1.*nm)))

structures = [slab]

from scipy.constants import c
freq0 = c/(5000.*nm)
wfreq0 = 2.*np.pi*freq0

pos_src = +500
pos_trs = +1000
#inc1    = fdtd.apply_direct_source('ex', ((0,0,pos_src),(-1,-1,pos_src)))
#inc2    = fdtd.apply_direct_source('hy', ((0,0,pos_src),(-1,-1,pos_src)))
fdtd.set_structures(structures)
print 'Setting Complete and READY to RUN'

plt.ion()

t0 = dt.now()
tc = 5000
ts = 500.
tmax = 200000 # 5 period for 1000 nm wavelength
#tmax = 10 # 5 period for 1000 nm wavelength
ex_ref = np.zeros(tmax, dtype=np.complex64)
#ex_trs = np.zeros(tmax, dtype=np.complex64)
for tstep in xrange(tmax):
    src1 = +min(1, tstep/100.)*np.exp(-1.j*wfreq0*(tstep+0.0)*fdtd.delta_t)
#    src2 = -min(1, tstep/100.)*np.exp(-1.j*wfreq0*(tstep+0.5)*fdtd.delta_t)
#    inc1.set_source(src1)
#    inc2.set_source(src2)

    fdtd.ex[:,:,pos_src] += src1
#    fdtd.ex[:,:,pos_src] = src1
    fdtd.updateE()
    fdtd.updateH()
    if tstep % 100 == 0:
        t1 = dt.now()
        print (fdtd.ex[:,:,pos_trs]*np.conjugate(fdtd.hy[:,:,pos_trs])).mean()
#        print('%s %05d time step\r' % (t1-t0, tstep)),
#        sys.stdout.flush()
        plt.clf()
        plt.plot(fdtd.ex.real[nx/2,ny/2,:], 'b')
        plt.plot(fdtd.hy.real[nx/2,ny/2,:], 'r')
        plt.ylim(-10., 10.)
        plt.draw()
