import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

import KEMP as SMS
from KEMP.mainfdtd import Basic_FDTD
from datetime import datetime as dtm

mode = '3D'
ds = 1.
nx, ny, nz = (400, 400, 400)
lx, ly, lz = (nx*ds, ny*ds, nz*ds)
x = np.ones(nx, dtype=np.float64)*ds
y = np.ones(ny, dtype=np.float64)*ds
z = np.ones(nz, dtype=np.float64)*ds
space_grid  = (x, y, z)
engine_name = 'nvidia_cuda'
device_ids = 0
fdtd     = Basic_FDTD(mode, space_grid , dtype=np.complex64, engine=engine_name, device_id=device_ids)#, MPI_extension='nonblock')

c = 299792458.
wavelen = 200
wfreq = SMS.to_NU(fdtd, 'angular frequency', 2.*np.pi*c/wavelen)
kx = 2.*np.pi/wavelen*np.cos(np.pi*.25)
ky = 2.*np.pi/wavelen*np.sin(np.pi*.25)

pml_apply = {'x':'', 'y':'', 'z':'+-'}
fdtd.apply_PML(pml_apply)

pbc_apply = {'x':True, 'y':True, 'z':False}
fdtd.apply_PBC(pbc_apply, klx=kx*lx, kly=0)

#fdtd_src = fdtd.apply_TFSF(((50,50,50),(-50,-50,-50)))
#fdtd_src.apply_PML(pml_apply)
#fdtd_src.apply_PBC(pbc_apply, klx=kx*lx, kly=0)

#inc = fdtd_src.apply_direct_source('Ex', ((0,0,30),(-1,-1,30)))

inc = fdtd.apply_TFSF1D('e', ((100, 100, 100), (-100, -100, -100)), (1./np.sqrt(2.), 0., 1./np.sqrt(2.)), (-1./np.sqrt(2.), 0., 1./np.sqrt(2.)))
#inc = fdtd.apply_TFSF1D('e', ((100, 100, 100), (-100, -100, -100)), (0., 0., 1.), (1., 0., 0.))

print 'Setting Complete and READY to RUN'

print fdtd.is_electric
print fdtd.is_magnetic
print fdtd.is_complex
print fdtd.is_uniform_grid

print fdtd.tfsfs[0].x_strt
print fdtd.tfsfs[0].y_strt
print fdtd.tfsfs[0].z_strt
print fdtd.tfsfs[0].x_stop
print fdtd.tfsfs[0].y_stop
print fdtd.tfsfs[0].z_stop

minv, maxv = -1e-3, +1e-3

if fdtd.master:
    plt.ion()
    fig = plt.figure(figsize=(12,12))
    pc00 = fig.add_subplot(331)
    pc01 = fig.add_subplot(332)
    pc02 = fig.add_subplot(333)
    pc10 = fig.add_subplot(334)
    pc11 = fig.add_subplot(335)
    pc12 = fig.add_subplot(336)
    pc20 = fig.add_subplot(337)
    pc21 = fig.add_subplot(338)
    pc22 = fig.add_subplot(339)



phase = kx*fdtd.x
tmax = 10000
errs = np.zeros(tmax, dtype=np.float32)
t0 = dtm.now()
for tstep in xrange(tmax):
    t1 = dtm.now()
#    src = min([tstep/1000., 1.)]*np.ones((fdtd.nx,fdtd.ny), dtype=np.complex64)*(np.exp(1.j*(phase-wfreq*fdtd.dt*tstep))[:,np.newaxis])
    src = min([tstep/1000., 1.])*np.exp(1.j*wfreq*fdtd.dt*tstep)
    inc.set_source(src)
    t2 = dtm.now()

    fdtd.updateE()

    t4 = dtm.now()

    fdtd.updateH()

    t3 = dtm.now()

    ey_x0_tot = fdtd.ey[+100+2,+100+2:-100-2,+100+2:-100-2]
    ez_x0_tot = fdtd.ez[+100+2,+100+2:-100-2,+100+2:-100-2]
    hy_x0_tot = fdtd.hy[+100+2,+100+2:-100-2,+100+2:-100-2]
    hz_x0_tot = fdtd.hz[+100+2,+100+2:-100-2,+100+2:-100-2]

    ez_y0_tot = fdtd.ez[+100+2:-100-2,+100+2,+100+2:-100-2]
    ex_y0_tot = fdtd.ex[+100+2:-100-2,+100+2,+100+2:-100-2]
    hz_y0_tot = fdtd.hz[+100+2:-100-2,+100+2,+100+2:-100-2]
    hx_y0_tot = fdtd.hx[+100+2:-100-2,+100+2,+100+2:-100-2]

    ex_z0_tot = fdtd.ex[+100+2:-100-2,+100+2:-100-2,+100+2]
    ey_z0_tot = fdtd.ey[+100+2:-100-2,+100+2:-100-2,+100+2]
    hx_z0_tot = fdtd.hx[+100+2:-100-2,+100+2:-100-2,+100+2]
    hy_z0_tot = fdtd.hy[+100+2:-100-2,+100+2:-100-2,+100+2]

    ey_x1_tot = fdtd.ey[-100-2,+100+2:-100-2,+100+2:-100-2]
    ez_x1_tot = fdtd.ez[-100-2,+100+2:-100-2,+100+2:-100-2]
    hy_x1_tot = fdtd.hy[-100-2,+100+2:-100-2,+100+2:-100-2]
    hz_x1_tot = fdtd.hz[-100-2,+100+2:-100-2,+100+2:-100-2]

    ez_y1_tot = fdtd.ez[+100+2:-100-2,-100-2,+100+2:-100-2]
    ex_y1_tot = fdtd.ex[+100+2:-100-2,-100-2,+100+2:-100-2]
    hz_y1_tot = fdtd.hz[+100+2:-100-2,-100-2,+100+2:-100-2]
    hx_y1_tot = fdtd.hx[+100+2:-100-2,-100-2,+100+2:-100-2]

    ex_z1_tot = fdtd.ex[+100+2:-100-2,+100+2:-100-2,-100-2]
    ey_z1_tot = fdtd.ey[+100+2:-100-2,+100+2:-100-2,-100-2]
    hx_z1_tot = fdtd.hx[+100+2:-100-2,+100+2:-100-2,-100-2]
    hy_z1_tot = fdtd.hy[+100+2:-100-2,+100+2:-100-2,-100-2]

    ey_x0_sca = fdtd.ey[+100-2,+100-2:-100+2,+100-2:-100+2]
    ez_x0_sca = fdtd.ez[+100-2,+100-2:-100+2,+100-2:-100+2]
    hy_x0_sca = fdtd.hy[+100-2,+100-2:-100+2,+100-2:-100+2]
    hz_x0_sca = fdtd.hz[+100-2,+100-2:-100+2,+100-2:-100+2]

    ez_y0_sca = fdtd.ez[+100-2:-100+2,+100-2,+100-2:-100+2]
    ex_y0_sca = fdtd.ex[+100-2:-100+2,+100-2,+100-2:-100+2]
    hz_y0_sca = fdtd.hz[+100-2:-100+2,+100-2,+100-2:-100+2]
    hx_y0_sca = fdtd.hx[+100-2:-100+2,+100-2,+100-2:-100+2]

    ex_z0_sca = fdtd.ex[+100-2:-100+2,+100-2:-100+2,+100-2]
    ey_z0_sca = fdtd.ey[+100-2:-100+2,+100-2:-100+2,+100-2]
    hx_z0_sca = fdtd.hx[+100-2:-100+2,+100-2:-100+2,+100-2]
    hy_z0_sca = fdtd.hy[+100-2:-100+2,+100-2:-100+2,+100-2]

    ey_x1_sca = fdtd.ey[-100+2,+100-2:-100+2,+100-2:-100+2]
    ez_x1_sca = fdtd.ez[-100+2,+100-2:-100+2,+100-2:-100+2]
    hy_x1_sca = fdtd.hy[-100+2,+100-2:-100+2,+100-2:-100+2]
    hz_x1_sca = fdtd.hz[-100+2,+100-2:-100+2,+100-2:-100+2]

    ez_y1_sca = fdtd.ez[+100-2:-100+2,-100+2,+100-2:-100+2]
    ex_y1_sca = fdtd.ex[+100-2:-100+2,-100+2,+100-2:-100+2]
    hz_y1_sca = fdtd.hz[+100-2:-100+2,-100+2,+100-2:-100+2]
    hx_y1_sca = fdtd.hx[+100-2:-100+2,-100+2,+100-2:-100+2]

    ex_z1_sca = fdtd.ex[+100-2:-100+2,+100-2:-100+2,-100+2]
    ey_z1_sca = fdtd.ey[+100-2:-100+2,+100-2:-100+2,-100+2]
    hx_z1_sca = fdtd.hx[+100-2:-100+2,+100-2:-100+2,-100+2]
    hy_z1_sca = fdtd.hy[+100-2:-100+2,+100-2:-100+2,-100+2]

    sx0_tot = ey_x0_tot*np.conjugate(hz_x0_tot) - ez_x0_tot*np.conjugate(hy_x0_tot)
    sy0_tot = ez_y0_tot*np.conjugate(hx_y0_tot) - ex_y0_tot*np.conjugate(hz_y0_tot)
    sz0_tot = ex_z0_tot*np.conjugate(hy_z0_tot) - ey_z0_tot*np.conjugate(hx_z0_tot)

    sx1_tot = ey_x1_tot*np.conjugate(hz_x1_tot) - ez_x1_tot*np.conjugate(hy_x1_tot)
    sy1_tot = ez_y1_tot*np.conjugate(hx_y1_tot) - ex_y1_tot*np.conjugate(hz_y1_tot)
    sz1_tot = ex_z1_tot*np.conjugate(hy_z1_tot) - ey_z1_tot*np.conjugate(hx_z1_tot)

    sx0_sca = ey_x0_sca*np.conjugate(hz_x0_sca) - ez_x0_sca*np.conjugate(hy_x0_sca)
    sy0_sca = ez_y0_sca*np.conjugate(hx_y0_sca) - ex_y0_sca*np.conjugate(hz_y0_sca)
    sz0_sca = ex_z0_sca*np.conjugate(hy_z0_sca) - ey_z0_sca*np.conjugate(hx_z0_sca)
                                                                                  
    sx1_sca = ey_x1_sca*np.conjugate(hz_x1_sca) - ez_x1_sca*np.conjugate(hy_x1_sca)
    sy1_sca = ez_y1_sca*np.conjugate(hx_y1_sca) - ex_y1_sca*np.conjugate(hz_y1_sca)
    sz1_sca = ex_z1_sca*np.conjugate(hy_z1_sca) - ey_z1_sca*np.conjugate(hx_z1_sca)

    s_tot = (abs(sx0_tot) + abs(sy0_tot) + abs(sz0_tot) + abs(sx1_tot) + abs(sy1_tot) + abs(sz1_tot)).sum()
    s_sca = (abs(sx0_sca) + abs(sy0_sca) + abs(sz0_sca) + abs(sx1_sca) + abs(sy1_sca) + abs(sz1_sca)).sum()

    if s_tot != 0.:
        errs[tstep] = s_sca/s_tot

    if tstep % 100 == 0:
        if fdtd.master:
            elapsed_time = dtm.now()-t0
            print '[%010d][%s][%s] elapsed time(per 1 tstep) of updateE: %s, updateH: %s, source: %s' \
                % (tstep, elapsed_time, t3-t1, t4-t2, t3-t4, t2-t1)
#                % (tstep, elapsed_time, t3-t1, fdtd.updateE_time, fdtd.updateH_time, t2-t1)

    if tstep % 100 == 0:
#        pdata00 = fdtd_src.ex.real[:,ny/2,:]
#        pdata01 = fdtd_src.ex.imag[:,ny/2,:]
#        pdata020 = fdtd_src.ex.real[nx/2,ny/2,:]
#        pdata021 = fdtd_src.ex.imag[nx/2,ny/2,:]
        pdata10  = fdtd    .ex.real[:,ny/2,:]
        pdata11  = fdtd    .ex.imag[:,ny/2,:]
        pdata120 = fdtd    .ex.real[nx/2,ny/2,:]
        pdata121 = fdtd    .ex.imag[nx/2,ny/2,:]
        if fdtd.master:
            pc02.cla(); pc12.cla(); pc22.cla();
            pc02.set_ylim(-1., +1.)
            pc12.set_ylim(-1., +1.)
            pc22.set_ylim(minv, maxv)
#            pc00.imshow(pdata00.T, vmax = 1., vmin=-1., cmap=plt.cm.RdBu_r, origin='lower')
#            pc01.imshow(pdata01.T, vmax = 1., vmin=-1., cmap=plt.cm.RdBu_r, origin='lower')
#            pc02.plot(pdata020, 'b')
#            pc02.plot(pdata021, 'r')
            pc10.imshow(pdata10.T, vmax = 1., vmin=-1., cmap=plt.cm.RdBu_r, origin='lower')
            pc11.imshow(pdata11.T, vmax = 1., vmin=-1., cmap=plt.cm.RdBu_r, origin='lower')
            pc12.plot(pdata120, 'b')
            pc12.plot(pdata121, 'r')
            pc20.imshow(pdata10.T, vmax = maxv, vmin=minv, cmap=plt.cm.RdBu_r, origin='lower')
            pc21.imshow(pdata11.T, vmax = maxv, vmin=minv, cmap=plt.cm.RdBu_r, origin='lower')
            pc22.plot(pdata120, 'b')
            pc22.plot(pdata121, 'r')
            plt.draw()


f = h5.File('1dtfsf_data.h5', 'w')
f.create_dataset('ex_real', data = pdata10)
f.create_dataset('ex_imag', data = pdata11)
f.create_dataset('errs', data=errs)
f.close()

