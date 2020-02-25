import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import sys
from KEMP import Basic_FDTD, Dielectric, to_epr, to_SI, to_NU, Gold, Silver, PA
from KEMP import structures as stc

nm = 1e-9
dx, dy, dz = [.1*nm, .1*nm, .1*nm]
nx, ny, nz = [2, 2, 5000] 
x = np.ones(nx, dtype=np.float64)*dx # 0 ~ 3000 nm
y = np.ones(ny, dtype=np.float64)*dy # 0 ~ 1 nm
z = np.ones(nz, dtype=np.float64)*dz # 0 ~ 1 nm
lx, ly, lz = dx*nx, dy*ny, dz*nz
space_grid = (x, y ,z)
engine_name = 'nvidia_cuda'
#engine_name = 'intel_cpu'
fdtd = Basic_FDTD('3D', space_grid, dtype=np.float32, engine=engine_name, device_id=2)

pml_apply = {'x':'' , 'y':''  , 'z':'+-'  }
pbc_apply = {'x':True, 'y':True, 'z':False}
fdtd.apply_PML(pml_apply)
fdtd.apply_PBC(pbc_apply)

slab_thick = 5.*nm
gap = 80.*nm

diel = Dielectric(to_epr(fdtd, n=1.5))
diep = Dielectric(to_epr(fdtd, n=10.**10))
#slab = stc.Box(diel, ((0.*nm,0.*nm,400.*nm),(lx,ly,400.*nm+slab_thick)))
#slab = stc.Box(Gold, ((0.*nm,0.*nm,400.*nm),(lx,ly,400.*nm+slab_thick)))
#slab = stc.Box(Silver, ((0.*nm,0.*nm,1000.*nm),(lx,ly,1000.*nm+slab_thick)))
slab_PA = stc.Box(PA  , ((-1.*nm,-1.*nm,400.*nm           ),(lx+1*nm,ly+1*nm,400.*nm+slab_thick    )))
slab_dl = stc.Box(diel, ((-1.*nm,-1.*nm,400.*nm+slab_thick),(lx+1*nm,ly+1*nm,400.*nm+slab_thick+gap)))
pecs = stc.Box(diep, ((-1.*nm,-1.*nm,400.*nm+slab_thick+gap),(lx+1*nm,ly+1*nm,lz+1*nm)))

#structures = [slab_PA]
structures = [slab_PA, slab_dl, pecs]

from scipy.constants import c
wavelength = np.arange(300., 801., 1.)*nm
freqs = c/wavelength
#freq0 = c/(650.*nm)
freq0 = c/(500.*nm)
wfreq0_NU = to_NU(fdtd, 'angular frequency', 2.*np.pi*freq0)

#kk = [0., 0., 1.]
#pp = [1., 0., 0.]
#inc = fdtd.apply_TFSF1D('e', ((1,1,5000),(-2,-2,-5000)), tuple(kk), tuple(pp), {'x':'+-', 'y':'+-', 'z':'+-'})

#rft_ex = fdtd.apply_RFT('ex', ((0,0,4000), (-1,-1,4000)), freqs)

#incfdtd = fdtd.apply_TFSF(((1,1,5000),(-2,-2,-5000)))
#pml_apply = {'x':'' , 'y':''  , 'z':'+-'  }
#pbc_apply = {'x':True, 'y':True, 'z':False}
#incfdtd.apply_PML(pml_apply)
#incfdtd.apply_PBC(pbc_apply)
#
pos_src = 3500
pos_trs = 6500
inc    = fdtd.apply_direct_source('ex', ((0,0,pos_src),(-1,-1,pos_src)))
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
tmax = 200000 # 5 period for 1000 nm wavelength
#tmax = 10 # 5 period for 1000 nm wavelength
ex_ref = np.zeros(tmax, dtype=np.complex64)
#ex_trs = np.zeros(tmax, dtype=np.complex64)
for tstep in xrange(tmax):
#    src = np.exp(-(tstep-tc)**2/(ts**2))*np.exp(1.j*wfreq0_NU*(tstep-tc)*fdtd.dt)
    src = np.exp(-(tstep-tc)**2/(ts**2))*np.cos(wfreq0_NU*(tstep-tc)*fdtd.dt)
    ex_ref[tstep] = fdtd.ex[:,:,pos_src].mean() - src
#    ex_trs[tstep] = fdtd.ex[:,:,pos_trs].mean()
    inc.set_source(src)
    fdtd.updateE()
    fdtd.updateH()
    if tstep % 5000 == 0:
        t1 = dt.now()
        print('%s %05d time step\r' % (t1-t0, tstep)),
        sys.stdout.flush()
        plt.clf()
        plt.plot(fdtd.ex.real[nx/2,ny/2,:], 'r')
#        plt.plot(fdtd.ex.imag[nx/2,ny/2,:], 'b')
        plt.ylim(-2., 2.)
        plt.draw()

#ex_trs_ft = rft_ex.export()

t1 = dt.now()
print t1-t0, 'Simulation Complete'
dt_SI = to_SI(fdtd, 'time', fdtd.dt)
tsteps = np.arange(tmax, dtype=fdtd.dtype)
t = tsteps*dt_SI
ex_src = np.exp(-(tsteps-tc)**2/(ts**2))*np.exp(1.j*wfreq0_NU*(tsteps-tc)*fdtd.dt)
ex_src_ft = ((dt_SI/(2.*np.pi))*ex_src[np.newaxis,:]*np.exp(-2.j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)

ex_ref_ft = ((dt_SI/(2.*np.pi))*ex_ref[np.newaxis,:]*np.exp(-2.j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)
ref_ft = abs(ex_ref_ft)**2/abs(ex_src_ft)**2

#ex_trs_ft = ((dt_SI/(2.*np.pi))*ex_trs[np.newaxis,:]*np.exp(-2.j*np.pi*freqs[:,np.newaxis]*t[np.newaxis,:])).sum(1)
#trs_ft = abs(ex_trs_ft)**2/abs(ex_src_ft)**2

#its_src_ft = abs(ex_src_ft)**2
#its_trs_ft = np.zeros_like(its_src_ft)
#its_trs_arr = (abs(ex_trs_ft)**2)
#for ii in xrange(its_trs_ft.size):
#    its_trs_ft[ii] = its_trs_arr[:,:,ii].mean()


from RTCalculator import TransferMatrix, IsoMaterial
from RTCalculator import Gold as tmmgold, Silver as tmmsilver, PAbs as tmm_pa

def epr_inc(wavelength): return 1.
def epr_mid(wavelength): return 1.5**2
def epr_tra(wavelength): return 1.
def epr_tra1(wavelength): return 1.e10

def mur_inc(wavelength): return 1.
def mur_mid(wavelength): return 1.
def mur_tra(wavelength): return 1.

inc_mat = IsoMaterial(epr_inc, mur_inc)
#mid_mat = tmmgold
#mid_mat = tmmsilver
#mid_mat = IsoMaterial(epr_mid, mur_mid)
mid_mat  = tmm_pa
mid_mat1 = IsoMaterial(epr_mid, mur_mid)
tra_mat  = IsoMaterial(epr_tra, mur_tra)
tra_mat1 = IsoMaterial(epr_tra1, mur_tra)

#tmm = TransferMatrix('E', inc_mat, tra_mat, [mid_mat], [slab_thick])
tmm = TransferMatrix('E', inc_mat, tra_mat1, [mid_mat, mid_mat1], [slab_thick, gap])
tmm.set_Wavelength(300.*nm, 801.*nm, 1.*nm)
tmm.set_IncAngleRad(0.)
tmm.calculation(['R'])
#tmm.calculation(['R', 'T'])
ref_tmm = tmm.results['R'][:,0]
#trs_tmm = tmm.results['T'][:,0]

wave_norm = wavelength/nm

plt.clf()
plt.ioff()
#plt.plot(wave_norm, its_trs_ft/its_src_ft, 'ro', label='FDTD simulation data')
plt.plot(wave_norm, ref_ft, 'ro', label='FDTD simulation data')
#plt.plot(wave_norm, trs_ft, 'bo', label='FDTD simulation data')
plt.plot(wave_norm, ref_tmm, 'r+', label='Theoretical data')
#plt.plot(wave_norm, trs_tmm, 'b+', label='Theoretical data')
plt.xlim(300, 800)
plt.ylim(  0,   1)
plt.legend()
plt.xlabel('wavelength (nm)')
plt.ylabel('R/T')

plt.savefig('RT.png')
plt.show()
