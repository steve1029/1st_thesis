import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import sys
from KEMP import Basic_FDTD, Dielectric, to_epr, to_SI, to_NU, Gold, Silver, PA
from KEMP import structures as stc

nm = 1e-9
dx, dy, dz = [.1*nm, .1*nm, .1*nm]
nx, ny, nz = [20000, 5, 5]
x = np.ones(nx, dtype=np.float64)*dx # 0 ~ 3000 nm
y = np.ones(ny, dtype=np.float64)*dy # 0 ~ 1 nm
z = np.ones(nz, dtype=np.float64)*dz # 0 ~ 1 nm
lx, ly, lz = dx*nx, dy*ny, dz*nz
space_grid = (x, y ,z)
#engine_name = 'nvidia_cuda'
engine_name = 'intel_cpu'
fdtd = Basic_FDTD('3D', space_grid, dtype=np.complex64, engine=engine_name, device_id=0)

pml_apply = {'x':'+-' , 'y':''  , 'z':''  }
pbc_apply = {'x':False, 'y':True, 'z':True}
fdtd.apply_PML(pml_apply)
fdtd.apply_PBC(pbc_apply)

slab_thick = 5.*nm
gap = 80*nm

diel = Dielectric(to_epr(fdtd, n=3.))
diep = Dielectric(to_epr(fdtd, n=10.**10))
#slab = stc.Box(diel, ((1000.*nm,0.*nm,0.*nm),(1000.*nm+slab_thick,ly,lz)))
slab = stc.Box(Gold, ((1000.*nm,0.*nm,0.*nm),(1000.*nm+slab_thick,ly,lz)))
#slab = stc.Box(PA  , ((1000.*nm,0.*nm,0.*nm),(1000.*nm+slab_thick,ly,lz)))
pecs = stc.Box(diep, ((1000.*nm+slab_thick+gap,0.*nm,0.*nm),(lx,ly,lz)))
#slab = stc.Box(Silver, ((1000.*nm,0.*nm,0.*nm),(1000.*nm+slab_thick,ly,lz)))
structures = [slab, pecs]
#structures = [slab]

from scipy.constants import c
wavelength = np.arange(300., 1001., 1.)*nm
freqs = c/wavelength
#freq0 = c/(650.*nm)
freq0 = c/(500.*nm)
wfreq0_NU = to_NU(fdtd, 'angular frequency', 2.*np.pi*freq0)

rft_ey = fdtd.apply_RFT('ey', ((-5000,0,0), (-5000,-1,-1)), freqs)
inc    = fdtd.apply_direct_source('ey', ((5000,0,0),(5000,-1,-1)))
fdtd.set_structures(structures)
print 'Setting Complete and READY to RUN'

#print fdtd.cpwd.dpolenum_max, fdtd.cpwd.cpolenum_max
#
#print fdtd.cpwd.my[:,:,:]
#print fdtd.cpwd.nx, fdtd.cpwd.ny, fdtd.cpwd.nz
#print fdtd.cpwd.mx.shape, fdtd.cpwd.my.shape, fdtd.cpwd.mz.shape
#
#print fdtd.cpwd.dr_cf_r[0][1:], fdtd.cpwd.dr_ce_r[0][1:]
#print fdtd.cpwd.cp_cf_r[0][1:], fdtd.cpwd.cp_cf_i[0][1:], fdtd.cpwd.cp_ce_r[0][1:], fdtd.cpwd.cp_ce_i[0][1:]
#print fdtd.cpwd.cp_cf_r[1][1:], fdtd.cpwd.cp_cf_i[1][1:], fdtd.cpwd.cp_ce_r[1][1:], fdtd.cpwd.cp_ce_i[1][1:]

plt.ion()

t0 = dt.now()
tc = 5000
ts = 500.
tmax = 50000 # 5 period for 1000 nm wavelength
for tstep in xrange(tmax):
    src = np.exp(-(tstep-tc)**2/(ts**2))*np.exp(1.j*wfreq0_NU*(tstep-tc)*fdtd.dt)
#    src = np.sin(wfreq0_NU*tstep*fdtd.dt)
    inc.set_source(src)
    fdtd.updateE()
    fdtd.updateH()
#    print abs(fdtd.cpwd.dr_fy_r[0][:,:,:]).sum(), \
#          abs(fdtd.cpwd.cp_fy_r[0][:,:,:]).sum(), abs(fdtd.cpwd.cp_fy_i[0][:,:,:]).sum(), \
#          abs(fdtd.cpwd.cp_fy_r[1][:,:,:]).sum(), abs(fdtd.cpwd.cp_fy_i[1][:,:,:]).sum()
    if tstep % 1000 == 0:
        t1 = dt.now()
        print('%s %05d time step\r' % (t1-t0, tstep)),
        sys.stdout.flush()
        plt.clf()
        plt.plot(fdtd.ey.real[:,ny/2,nz/2], 'r')
        plt.plot(fdtd.ey.imag[:,ny/2,nz/2], 'b')
#        plt.plot(fdtd.cpwd.dr_fy_r[0][:,ny/2,nz/2], 'g')
#        plt.plot(fdtd.cpwd.cp_fy_r[0][:,ny/2,nz/2], 'b')
#        plt.plot(fdtd.cpwd.cp_fy_r[1][:,ny/2,nz/2], 'k')
        plt.ylim(-2., 2.)
        plt.draw()
#        plt.show()

ey_trs_ft = rft_ey.export()

t1 = dt.now()
print t1-t0, 'Simulation Complete'
dt_SI = to_SI(fdtd, 'time', fdtd.dt)
tsteps = np.arange(tmax, dtype=fdtd.dtype)
t = tsteps*dt_SI
ey_src = np.exp(-(tsteps-tc)**2/(ts**2))*np.exp(1.j*wfreq0_NU*(tsteps-tc)*fdtd.dt)
ey_src_ft = ((dt_SI/(2.*np.pi))*ey_src[np.newaxis,:]*np.exp(-2.j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)

its_src_ft = abs(ey_src_ft)**2
its_trs_ft = np.zeros_like(its_src_ft)
its_trs_arr = (abs(ey_trs_ft)**2)
for ii in xrange(its_trs_ft.size):
    its_trs_ft[ii] = its_trs_arr[:,:,ii].mean()


from RTCalculator import TransferMatrix, IsoMaterial
from RTCalculator import Gold as tmmgold, Silver as tmmsilver

def epr_inc(wavelength): return 1.
def epr_mid(wavelength): return 3.**2
def epr_tra(wavelength): return 1.

def mur_inc(wavelength): return 1.
def mur_mid(wavelength): return 1.
def mur_tra(wavelength): return 1.

inc_mat = IsoMaterial(epr_inc, mur_inc)
mid_mat = tmmgold
#mid_mat = tmmsilver
#mid_mat = IsoMaterial(epr_mid, mur_mid)
tra_mat = IsoMaterial(epr_tra, mur_tra)

tmm = TransferMatrix('E', inc_mat, tra_mat, [mid_mat], [slab_thick])
tmm.set_Wavelength(300.*nm, 1000.*nm, 1.*nm)
tmm.set_IncAngleRad(0.)
tmm.calculation(['T'])
trs_tmm = tmm.results['T'][:,0]

wave_norm = wavelength/nm

plt.clf()
plt.ioff()
plt.plot(wave_norm, its_trs_ft/its_src_ft, 'ro', label='FDTD simulation data')
plt.plot(wave_norm, trs_tmm, 'b+', label='Theoretical data')
plt.xlim(300, 1000)
plt.ylim(  0,    1)
plt.legend()
plt.xlabel('wavelength (nm)')
plt.ylabel('Transmittance')
plt.show()
