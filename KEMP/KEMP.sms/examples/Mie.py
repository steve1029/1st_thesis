#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0

import GBES
from GBES.materials import *
from GBES.structures import Sphere
from datetime import datetime as dat

space_dim = '3D'
nm = 1.e-9
grid_size = 5*nm
dx, dy, dz = [grid_size for i in xrange(3)]
nx, ny, nz = 400, 400, 400
comp_arch = 'nvidia_cuda' # computing architecture
dev_num = 0 # number of gpu device. if your computer has only 1 gpu, the number must be 0.

# in this case, we use 10 nm uniform grid structure. if you want, you can also use nonuniform-grid structures.
x = np.ones(nx, dtype=np.float64)*grid_size  #( length of x-axis: 3000 nm )
y = np.ones(ny, dtype=np.float64)*grid_size  #( length of x-axis: 4000 nm )
z = np.ones(nz, dtype=np.float64)*grid_size  #( length of x-axis: 5000 nm )
grid_structure = (x, y, z)
lx, ly, lz = nx*dx, ny*dy, nz*dz

# Main FDTD space
fdtdspace = GBES.Basic_FDTD(space_dim, grid_structure, dtype=np.float32, engine=comp_arch, device_id=0, MPI_extension=False)

# Boundary Conditions
pml_boundary = {'x':'+-', 'y':'+-', 'z':'+-'}
pbc_boundary = {'x':False, 'y':False, 'z':False}
fdtdspace.apply_PML(pml_boundary)
fdtdspace.apply_PBC(pbc_boundary)

tfsf_boundary = {'x':'+-', 'y':'+-', 'z':'+-'}
tfsf_region = ((150,150,150), (-150,-150,-150))
tfsf_fdtd = fdtdspace.apply_TFSF(tfsf_region, tfsf_boundary, is_oblique=False)
tfsf_pml_boundary = {'x':'+-', 'y':'', 'z':''}
tfsf_pbc_boundary = {'x':False, 'y':True, 'z':True}
tfsf_fdtd.apply_PML(tfsf_pml_boundary)
tfsf_fdtd.apply_PBC(tfsf_pbc_boundary)

# Incident sources
src_region = ((50,0,0),(50,-1,-1))
#src = fdtdspace.apply_direct_source('ez', src_region)
tfsf_src = tfsf_fdtd.apply_direct_source('ez', src_region)
wavelength  = 500.*nm
wavelengths = np.arange(100., 1000., 10.,)*nm
wfreq0 = 2.*np.pi*c0/wavelength
freqs = c0/wavelengths
pulse_width = 1000.

#tmax = 30000
#tstep0 = 10000
#tsteps = np.arange(tmax)
#ez_src = np.exp(-.5*((tsteps-tstep0)/pulse_width)**2)*np.cos(tsteps*fdtdspace.delta_t*wfreq0)
#plt.plot(tsteps, ez_src)
#plt.show()
#import sys
#sys.exit()

# Setting running fourier transform
fdata_ey_x0 = fdtdspace.apply_RFT('ey', ((+100, +100, +100), (+100, -100, -100)), freqs)
fdata_ez_x0 = fdtdspace.apply_RFT('ez', ((+100, +100, +100), (+100, -100, -100)), freqs)
fdata_ey_x1 = fdtdspace.apply_RFT('ey', ((-100, +100, +100), (-100, -100, -100)), freqs)
fdata_ez_x1 = fdtdspace.apply_RFT('ez', ((-100, +100, +100), (-100, -100, -100)), freqs)
                                                                              
fdata_ez_y0 = fdtdspace.apply_RFT('ez', ((+100, +100, +100), (-100, +100, -100)), freqs)
fdata_ex_y0 = fdtdspace.apply_RFT('ex', ((+100, +100, +100), (-100, +100, -100)), freqs)
fdata_ez_y1 = fdtdspace.apply_RFT('ez', ((+100, -100, +100), (-100, -100, -100)), freqs)
fdata_ex_y1 = fdtdspace.apply_RFT('ex', ((+100, -100, +100), (-100, -100, -100)), freqs)
                                                                              
fdata_ex_z0 = fdtdspace.apply_RFT('ex', ((+100, +100, +100), (-100, -100, +100)), freqs)
fdata_ey_z0 = fdtdspace.apply_RFT('ey', ((+100, +100, +100), (-100, -100, +100)), freqs)
fdata_ex_z1 = fdtdspace.apply_RFT('ex', ((+100, +100, -100), (-100, -100, -100)), freqs)
fdata_ey_z1 = fdtdspace.apply_RFT('ey', ((+100, +100, -100), (-100, -100, -100)), freqs)

fdata_hy_x0 = fdtdspace.apply_RFT('hy', ((+100, +100, +100), (+100, -100, -100)), freqs)
fdata_hz_x0 = fdtdspace.apply_RFT('hz', ((+100, +100, +100), (+100, -100, -100)), freqs)
fdata_hy_x1 = fdtdspace.apply_RFT('hy', ((-100, +100, +100), (-100, -100, -100)), freqs)
fdata_hz_x1 = fdtdspace.apply_RFT('hz', ((-100, +100, +100), (-100, -100, -100)), freqs)

fdata_hz_y0 = fdtdspace.apply_RFT('hz', ((+100, +100, +100), (-100, +100, -100)), freqs)
fdata_hx_y0 = fdtdspace.apply_RFT('hx', ((+100, +100, +100), (-100, +100, -100)), freqs)
fdata_hz_y1 = fdtdspace.apply_RFT('hz', ((+100, -100, +100), (-100, -100, -100)), freqs)
fdata_hx_y1 = fdtdspace.apply_RFT('hx', ((+100, -100, +100), (-100, -100, -100)), freqs)

fdata_hx_z0 = fdtdspace.apply_RFT('hx', ((+100, +100, +100), (-100, -100, +100)), freqs)
fdata_hy_z0 = fdtdspace.apply_RFT('hy', ((+100, +100, +100), (-100, -100, +100)), freqs)
fdata_hx_z1 = fdtdspace.apply_RFT('hx', ((+100, +100, -100), (-100, -100, -100)), freqs)
fdata_hy_z1 = fdtdspace.apply_RFT('hy', ((+100, +100, -100), (-100, -100, -100)), freqs)

# Setting Structures
ref_n = 3.
material = Dielectric(epr=ref_n**2)
#material = Dimagnetic(mur=4.)
#material = Dielectromagnetic(epr=4., mur=8.)
#material = gold
#material = silver
radius = 75.*nm
dielec_box = Sphere(material, (lx*.5, ly*.5, lz*.5), radius)
structures = [dielec_box]
fdtdspace.set_structures(structures)

# display
plt.ion()

# Running time loop
tmax = 30000
tstep0 = 10000
time0 = dat.now()
for tstep in xrange(tmax):
    source = np.exp(-.5*((tstep-tstep0)/pulse_width)**2)*np.cos(tstep*fdtdspace.delta_t*wfreq0)
#    source = np.sin(tstep*fdtdspace.delta_t*wfreq0)
#    src.set_source(source)
    tfsf_src.set_source(source)

    fdtdspace.updateE()
    fdtdspace.updateH()
    if tstep % 500 == 0:
        plt.imshow(fdtdspace.ez[:,:,nz/2].T, vmin=-1., vmax=+1., origin='lower', cmap=plt.cm.RdBu_r)
        plt.draw()
        print tstep, source, dat.now()-time0

tsteps = np.arange(tmax)
ez_src = np.exp(-.5*((tsteps-tstep0)/pulse_width)**2)*np.cos(tsteps*fdtdspace.delta_t*wfreq0)
ft_src = ((fdtdspace.delta_t/(2.*np.pi))*ez_src[np.newaxis,:]*np.exp(-2.j*np.pi*freqs[:,np.newaxis]*fdtdspace.delta_t*tsteps[np.newaxis,:])).sum(1)

ey_x0 = fdata_ey_x0.export() 
ez_x0 = fdata_ez_x0.export()
ey_x1 = fdata_ey_x1.export()
ez_x1 = fdata_ez_x1.export()
        
ez_y0 = fdata_ez_y0.export()
ex_y0 = fdata_ex_y0.export()
ez_y1 = fdata_ez_y1.export()
ex_y1 = fdata_ex_y1.export()
        
ex_z0 = fdata_ex_z0.export()
ey_z0 = fdata_ey_z0.export()
ex_z1 = fdata_ex_z1.export()
ey_z1 = fdata_ey_z1.export()
        
hy_x0 = fdata_hy_x0.export()
hz_x0 = fdata_hz_x0.export()
hy_x1 = fdata_hy_x1.export()
hz_x1 = fdata_hz_x1.export()
       
hz_y0 = fdata_hz_y0.export()
hx_y0 = fdata_hx_y0.export()
hz_y1 = fdata_hz_y1.export()
hx_y1 = fdata_hx_y1.export()
       
hx_z0 = fdata_hx_z0.export()
hy_z0 = fdata_hy_z0.export()
hx_z1 = fdata_hx_z1.export()
hy_z1 = fdata_hy_z1.export()

sca_x0 = - ey_x0*hz_x0 + ez_x0*hy_x0
sca_x1 = + ey_x1*hz_x1 - ez_x1*hy_x1

sca_y0 = - ez_y0*hx_y0 + ex_y0*hz_y0
sca_y1 = + ez_y1*hx_y1 - ex_y1*hz_y1

sca_z0 = - ex_z0*hy_z0 + ey_z0*hx_z0
sca_z1 = + ex_z1*hy_z1 - ey_z1*hx_z1

#sca_x0 = abs(ey_x0)**2 + abs(ez_x0)**2
#sca_x1 = abs(ey_x1)**2 + abs(ez_x1)**2

#sca_y0 = abs(ez_y0)**2 + abs(ex_y0)**2
#sca_y1 = abs(ez_y1)**2 + abs(ex_y1)**2

#sca_z0 = abs(ex_z0)**2 + abs(ey_z0)**2
#sca_z1 = abs(ex_z1)**2 + abs(ey_z1)**2

sca = sca_x0.sum(0).sum(0) + sca_x1.sum(0).sum(0) \
    + sca_y0.sum(0).sum(0) + sca_y1.sum(0).sum(0) \
    + sca_z0.sum(0).sum(0) + sca_z1.sum(0).sum(0)

norm_sca = sca/abs(ft_src)**2

plt.ioff()
plt.clf()
plt.plot(wavelengths/nm, norm_sca.real, 'red')
plt.plot(wavelengths/nm, norm_sca.imag, 'blue')
plt.plot(wavelengths/nm, abs(norm_sca), 'black')
plt.savefig('Mie_scattering.png')
plt.show()

print 'Simulation END'
