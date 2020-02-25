import numpy as np
from scipy.constants import c as c0
import GBES
from GBES.materials import Dielectric, Dimagnetic, Dielectromagnetic, gold, silver
from GBES.structures import Box

space_dim = '3D'
nm = 1.e-9
grid_size = 10*nm
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
pml_boundary = {'x':'+-', 'y':'+-', 'z':''}
pbc_boundary = {'x':False, 'y':False, 'z':True}
tfsf_boundary = {'x':'+-', 'y':'+-', 'z':''}
tfsf_region = ((100,100,100), (-100,-100,-100))
fdtdspace.apply_PML(pml_boundary)
fdtdspace.apply_PBC(pbc_boundary)
tfsf_fdtd = fdtdspace.apply_TFSF(tfsf_region, tfsf_boundary, is_oblique=False)

# Incident sources
src_region = ((50,0,0),(50,-1,-1))
#src = fdtdspace.apply_direct_source('ez', src_region)
#src = fdtdspace.apply_monochromatic_source('ez', src_region)
tfsf_src = tfsf_fdtd.apply_direct_source('ez', src_region)
wavelength = 500.*nm
wfreq = 2.*np.pi*c0/wavelength
wfreq_NU = GBES.to_NU(fdtdspace, 'angular frequency', wfreq)

# Setting Structures
material = Dielectric(epr=10.e10)
#material = Dimagnetic(mur=4.)
#material = Dielectromagnetic(epr=4., mur=8.)
#material = gold
#material = silver
box_size = (1000*nm, 1000*nm, 1000*nm)
dielec_box = Box(material, (((lx-box_size[0])*.5, (ly-box_size[1])*.5, (lz-box_size[2])*.5), \
                            ((lx+box_size[0])*.5, (ly+box_size[1])*.5, (lz+box_size[2])*.5)) ) 
structures = [dielec_box]
fdtdspace.set_structures(structures)

# display
import matplotlib.pyplot as plt
plt.ion()

# Running time loop
tmax = 10000
for tstep in xrange(tmax):
    source     = np.sin(    tstep*fdtdspace.dt*wfreq_NU)
#   obq_source = np.exp(1.j*tstep*fdtdspace.dt*wfreq_NU) # Must be complex
#   src.set_source(source)
    tfsf_src.set_source(source)

    fdtdspace.updateE()
    fdtdspace.updateH()
    if tstep % 100 == 0:
        print tstep 
        plt.imshow(fdtdspace.ez[:,:,nz/2].T, vmin=-1., vmax=+1., origin='lower', cmap=plt.cm.RdBu_r)
        plt.draw()

print 'Simulation END'
