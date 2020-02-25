#!/usr/bin/env python
import os, time, datetime, sys, psutil
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import source, space, plotfield, structure
from scipy.constants import c

#------------------------------------------------------------------#
#----------------------- Paramter settings ------------------------#
#------------------------------------------------------------------#

savedir = '/home/ldg/script/pyctypes/HPF.cfft.diel.CPML.MPI/'

nm = 1e-9
um = 1e-6

Lx, Ly, Lz = 128*2*um, 128*2*um, 128*2*um
Nx, Ny, Nz = 128, 16, 128
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

courant = 1./4
dt = courant * min(dx,dy,dz) / c
Tstep = 2001

wv_srt = 40*um
wv_end = 50*um
interval = 0.1*um
spread   = 0.3
pick_pos = 1000
plot_per = 100

# Set the type of input source.
#Src = source.Gaussian(dt, dtype=np.float64)
#Src.wvlen([wv_srt, wv_end, interval, spread])
Src = source.Sine(dt, np.float64)
Src.set_wvlen( 50 * um)

src_xpos = 20

#------------------------------------------------------------------#
#-------------------------- Call objects --------------------------#
#------------------------------------------------------------------#

# Space
Space = space.Space((Nx, Ny, Nz), (dx, dy, dz), courant, dt, Tstep, np.float64)

# Slab
Box1_srt = (Space.Nxc-15, 0, Space.Nzc-15)
Box1_end = (Space.Nxc+15, Space.Ny, Space.Nzc+15)
Box = structure.Box(Space, Box1_srt, Box1_end, 1e6, 1.)
#Box = structure.Box(Space, Box2_srt, Box2_end, 1. , 1.)
#Box = structure.Box(Space, Box3_srt, Box3_end, 4. , 1.)
#Box = structure.Box(Space, Box3_srt, Box3_end, 1e6 , 1.)

# Set PML
#Space.set_PML({'x':'+-', 'y':'+-', 'z':'+-'}, 10)
Space.set_PML({'x':'+-', 'y':'', 'z':'+-'}, 10)
#Space.set_PML({'x':'+-', 'y':'', 'z':''}, 10)
#Space.set_PML({'x':'', 'y':'+-', 'z':''}, 10)
#Space.set_PML({'x':'', 'y':'', 'z':''}, 10)

# Save eps, mu and PML data.
#Space.save_PML_parameters('./')
#Space.save_eps_mu(savedir)

# Plain wave propagating along x axis.
Space.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Space.Ny, Space.Nz))

# Line source along x axis.
#Space.set_src_pos((1, Space.Nyc, Space.Nzc), (Space.Nx, Space.Nyc+1, Space.Nzc+1))

# Line source along y axis.
#Space.set_src_pos((src_xpos, 0, Space.Nzc), (src_xpos+1, Space.Ny, Space.Nzc+1))

# Line source along z axis.
#Space.set_src_pos((src_xpos, 0, Space.Nzc), (src_xpos+1, Space.Ny, Space.Nzc+1))

# Set plotfield options
graphtool = plotfield.Graphtool(Space, savedir)

# Initialize the core
Space.init_update_equations(turn_on_omp=True)

# Save what time the simulation begins.
start_time = datetime.datetime.now()

# time loop begins
for tstep in range(Space.tsteps):

	# At the start point
	if tstep == 0:
		Space.MPIcomm.Barrier()
		if Space.MPIrank == 0:
			print("Total time step: %d" %(Space.tsteps))
			print(("Size of a total field array : %05.2f Mbytes" %(Space.TOTAL_NUM_GRID_SIZE)))
			print("Simulation start: {}".format(datetime.datetime.now()))
		
	pulse_re = Src.pulse_re(tstep, pick_pos=pick_pos)
	pulse_im = Src.pulse_im(tstep, pick_pos=pick_pos)

	#Space.put_src('Ex_re', 'Ex_im', pulse_re, 0, 'soft')
	Space.put_src('Ey_re', 'Ey_im', pulse_re, 0, 'soft')
	#Space.put_src('Ez_re', 'Ez_im', pulse_re, 0, 'soft')

	Space.updateH(tstep)
	Space.updateE(tstep)

	# Plot the field profile
	if tstep % plot_per == 0:
		#graphtool.plot2D3D('Ex', tstep, yidx=Space.Nyc, colordeep=0.1, stride=2, zlim=.1)
		graphtool.plot2D3D('Ey', tstep, xidx=Space.Nxc, colordeep=2, stride=2, zlim=2)
		#graphtool.plot2D3D('Ey', tstep, yidx=Space.Nyc, colordeep=2, stride=2, zlim=2)
		#graphtool.plot2D3D('Ey', tstep, zidx=Space.Nzc, colordeep=2, stride=2, zlim=2)
		#graphtool.plot2D3D('Ez', tstep, yidx=Space.Nyc, colordeep=0.1, stride=2, zlim=.1)
		#graphtool.plot2D3D('Hx', tstep, xidx=Space.Nxc, colordeep=1e-3, stride=2, zlim=1e-3)
		#graphtool.plot2D3D('Hy', tstep, xidx=Space.Nxc, colordeep=1e-3, stride=2, zlim=1e-3)
		#graphtool.plot2D3D('Hz', tstep, xidx=Space.Nxc, colordeep=1e-3, stride=2, zlim=1e-3)

		if Space.MPIrank == 0:

			interval_time = datetime.datetime.now()
			print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/Space.tsteps)))

if Space.MPIrank == 0:

	# Simulation finished time
	finished_time = datetime.datetime.now()

	# Record simulation size and operation time
	if not os.path.exists("./record") : os.mkdir("./record")
	record_path = "./record/record_%s.txt" %(datetime.date.today())

	if not os.path.exists(record_path):
		f = open( record_path,'a')
		f.write("{:4}\t{:4}\t{:4}\t{:4}\t{:4}\t\t{:4}\t\t{:4}\t\t{:8}\t{:4}\t\t\t\t{:12}\t{:12}\n\n" \
			.format("Node","Nx","Ny","Nz","dx","dy","dz","tsteps","Time","VM/Node(GB)","RM/Node(GB)"))
		f.close()

	me = psutil.Process(os.getpid())
	me_rssmem_GB = float(me.memory_info().rss)/1024/1024/1024
	me_vmsmem_GB = float(me.memory_info().vms)/1024/1024/1024

	cal_time = finished_time - start_time
	f = open( record_path,'a')
	f.write("{:2d}\t\t{:04d}\t{:04d}\t{:04d}\t{:5.2e}\t{:5.2e}\t{:5.2e}\t{:06d}\t\t{}\t\t{:06.3f}\t\t\t{:06.3f}\n" \
				.format(Space.MPIsize, Space.Nx, Space.Ny, Space.Nz,\
					Space.dx, Space.dy, Space.dz, Space.tsteps, cal_time, me_vmsmem_GB, me_rssmem_GB))
	f.close()
	
	print("Simulation finished: {}".format(datetime.datetime.now()))
