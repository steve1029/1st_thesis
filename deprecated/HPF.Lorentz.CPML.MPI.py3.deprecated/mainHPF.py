#!/usr/bin/env python

import source, space, plotting, structure, models
import os, time, datetime, sys, psutil
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

nm = 1e-9
um = 1e-6

# Create Space object.

#------------------------------------------------------------------------------------------------#
#------------------------------------------- For 6 nodes ----------------------------------------#
#------------------------------------------------------------------------------------------------#

#Space = space.Space(( 840,  128,  128), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  128,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  256,  256), (30*nm, 60*nm, 60*nm), 2048, np.float64, courant=1./4)
#Space = space.Space((1008,  256,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  256,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  512,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1008,  512,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 1008 MB
#Space = space.Space((1176,  512,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 1176 MB
#Space = space.Space((1344,  512,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1512,  512,  256), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1008,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 2016 MB
#Space = space.Space((1176,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 2688 MB
#Space = space.Space((1512,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 3024 MB
#Space = space.Space((1680,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 3360 MB
#Space = space.Space((1848,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # 3696 MB
#Space = space.Space((2016,  512,  512), (30*nm, 60*nm, 60*nm), 1001, np.float64, courant=1./4) # fail


#Space = space.Space(( 840,   64,   64), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,   64,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  128,  128), (30*nm, 120*nm, 120*nm), 1024, np.float64, courant=1./4)
#Space = space.Space((1008,  128,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  128,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  256,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2520,  128,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1008,  256,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1176,  256,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  256,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1512,  256,  128), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1008,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1176,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1512,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1680,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1848,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2016,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4) # 1008 MB
#Space = space.Space((2520,  256,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2520,  512,  256), (30*nm, 120*nm, 120*nm), 1001, np.float64, courant=1./4)


#Space = space.Space(( 840,   32,   32), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,   32,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,   64,   64), (30*nm, 240*nm, 240*nm), 1024, np.float64, courant=1./4)
#Space = space.Space((1008,   64,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,   64,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  128,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1008,  128,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1176,  128,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  128,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1512,  128,   64), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space(( 840,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1008,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1176,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1344,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1512,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1680,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((1848,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2016,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2520,  128,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2520,  256,  128), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((2520,  256,  256), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((3024,  256,  256), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((3024,  256,  512), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)
#Space = space.Space((3864,  256,  512), (30*nm, 240*nm, 240*nm), 1001, np.float64, courant=1./4)

#------------------------------------------------------------------------------------------------#
#------------------------------------------- For 8 nodes ----------------------------------------#
#------------------------------------------------------------------------------------------------#

#Space = space.Space(( 32,  32,  32), (30*nm, 30*nm, 30*nm), 5001, np.float64, courant=1./4) # 16 MB
#Space = space.Space(( 64,  64,  64), (30*nm, 30*nm, 30*nm), 5001, np.float64, courant=1./4) # 16 MB
#Space = space.Space(( 256,   32,   32), (30*nm, 30*nm, 30*nm), 5001, np.float64, courant=1./4) # 16 MB
Space = space.Space(( 128,  128,  128), (30*nm, 30*nm, 30*nm), 2001, np.float64, courant=1./4) # 16 MB

#Space = space.Space(( 512,  256,  256), (30*nm, 60*nm, 60*nm), 1024, np.float64, courant=1./4) # fail
#Space = space.Space((1024,  512,  256), (30*nm, 60*nm, 60*nm), 2048, np.float64, courant=1./4) # fail
#Space = space.Space((1024,  512,  512), (30*nm, 60*nm, 60*nm), 2048, np.float64, courant=1./4) # fail
#Space = space.Space((2048,  512,  512), (30*nm, 60*nm, 60*nm), 4096, np.float64, courant=1./4) # fail

#Space = space.Space(( 128,  128,  128), (120*nm, 120*nm, 120*nm),  800, np.float64, courant=1./4)
#Space = space.Space(( 512,  128,  128), ( 30*nm, 120*nm, 120*nm),  512, np.float64, courant=1./4)
#Space = space.Space((1024,  256,  128), ( 30*nm, 120*nm, 120*nm), 2048, np.float64, courant=1./4)
#Space = space.Space((2048,  256,  256), ( 30*nm, 120*nm, 120*nm), 4096, np.float64, courant=1./4)
#Space = space.Space((2048,  512,  256), ( 30*nm, 120*nm, 120*nm), 4096, np.float64, courant=1./4)
#Space = space.Space((2048,  512,  512), ( 30*nm, 120*nm, 120*nm), 4096, np.float64, courant=1./4)

#Space = space.Space(( 128, 128, 128), (240*nm, 240*nm, 240*nm), 2048, np.float64, courant=1./4)
#Space = space.Space(( 256, 256, 256), (240*nm, 240*nm, 240*nm),  512, np.float64, courant=1./4)
#Space = space.Space((1024, 128,  64), ( 30*nm, 240*nm, 240*nm), 2048, np.float64, courant=1./4)
#Space = space.Space((1024, 128, 128), ( 30*nm, 240*nm, 240*nm), 2048, np.float64, courant=1./4)
#Space = space.Space((2048, 128, 128), ( 30*nm, 240*nm, 240*nm), 4096, np.float64, courant=1./4)
#Space = space.Space((2048, 256, 128), ( 30*nm, 240*nm, 240*nm), 4096, np.float64, courant=1./4)
#Space = space.Space((2048, 256, 256), ( 30*nm, 240*nm, 240*nm), 4096, np.float64, courant=1./4)
#Space = space.Space((4096, 256, 256), ( 30*nm, 240*nm, 240*nm), 8192, np.float64, courant=1./4)
#Space = space.Space((4096, 512, 256), ( 30*nm, 240*nm, 240*nm), 8192, np.float64, courant=1./4)

# Set PML
Space.set_PML({'x':'+-', 'y':'', 'z':'+-'}, 10)
Space.save_PML_parameters('./')

# Set Structures and materials
#Struct1		 = structure.Box(Space, [78,54,54], [104,74,74])
#Silver_Palik = models.Lorentz3P(Struct1, 0, 1., [7.556, 0,0], [-0.618169,0,0], [0.035948,0,0])
#sys.exit()

# Set source
Src = source.Gaussian(Space.dt, Space.dtype)
Src.wvlen([600*nm, 2200*nm, .1*nm, 0.25])

#Src = source.Sine(Space.dt, Space.dtype)
#Src.set_wvlen( 600 * nm)

src_xpos = Space.Nxc+20
#src_xpos = 15

#Space.set_src_pos((src_xpos, 0, 0), (src_xpos+1, Space.Ny, Space.Nz))
#Space.set_src_pos((src_xpos, 0, Space.Nzc), (src_xpos+1, Space.Ny, Space.Nzc+1))
#Space.set_src_pos((src_xpos, 0, Space.Nzc-15), (src_xpos+1, Space.Ny, Space.Nzc+15))

#Space.set_src_pos((src_xpos, Space.Nyc, 0), (src_xpos+1, Space.Nyc+1, Space.Nz))
Space.set_src_pos((src_xpos, 0, Space.Nzc-20), (src_xpos+1, Space.Nyc, Space.Nzc))
#Space.set_src_pos((src_xpos, Space.Nyc-20, 0), (src_xpos+1, Space.Nyc, Space.Nz))

#Space.set_src_pos((src_xpos, Space.Nyc, Space.Nzc), (src_xpos+1, Space.Nyc+1, Space.Nzc+1))
#Space.set_src_pos((src_xpos, Space.Nyc, Space.Nzc-15), (src_xpos+1, Space.Nyc+1, Space.Nzc+15))
#Space.set_src_pos((src_xpos, Space.Nyc, Space.Nzc), (src_xpos+1, Space.Nyc+1, Space.Nzc+1))

# Set plotting options
savedir = '/home/ldg/script/pyctypes/HPF.Lorentz.CPML.MPI.py3/'
graphtool = plotting.Graphtool(Space, savedir)

# initialize the core
Space.init_update_equations(turn_on_omp=True)

start_time = datetime.datetime.now()
# time loop begins
for tstep in range(Space.tsteps):

	# At the start point
	if tstep == 0:
		Space.comm.Barrier()
		if Space.MPIrank == 0:
			print("Total time step: %d" %(Space.tsteps))
			print(("Size of a total field array : %05.2f Mbytes" %(Space.Mbytes_of_totalSIZE)))
			print("Simulation start")
		
	pulse_re = Src.pulse_re(tstep, pick_pos=500)
	pulse_im = Src.pulse_im(tstep, pick_pos=500)

	#Space.put_src('Ey_re', 'Ey_im', pulse_re, pulse_im, 'soft')
	Space.put_src('Ey_re', 'Ey_im', pulse_re, 0, 'soft')
	#Space.put_src('Ez_re', 'Ez_im', pulse_re, 0, 'soft')
	Space.updateH(tstep)
	Space.updateE(tstep)

	Space.get_src(tstep)

	# Plot the field profile
	if tstep % 100 == 0:
		graphtool.plot2D3D('Ey', tstep, yidx=Space.Nyc, colordeep=0.5, stride=2, zlim=4.)
		#graphtool.plot2D3D('Ez', tstep, zidx=Space.Nzc, colordeep=0.5, stride=2, zlim=4.)

		if Space.MPIrank == 0:

			interval_time = datetime.datetime.now()
			print(("time: %s, step: %05d, %5.2f%%" %(interval_time-start_time, tstep, 100.*tstep/Space.tsteps)))

	# At the last point
	if tstep == (Space.tsteps-1):
		
		# Wait until every nodes finished their job.
		Space.comm.Barrier()

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
			
			# Print simulation time
			print(("time: %s, step: %05d, %5.2f%%" %(cal_time, tstep, 100.*tstep/Space.tsteps)))
			print("Simulation finished")
			print("Plotting Start")

		Space.comm.Barrier()

graphtool.plot_src(Src)

if Space.MPIrank == 0 : 

	plot_finished_time = datetime.datetime.now()
	print("time: %s" %(plot_finished_time - start_time))
	print("Plotting finished")
