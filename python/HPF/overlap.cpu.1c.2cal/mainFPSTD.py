#!/usr/bin/env python

import source, Structure, PML, plotting, build
import os, time, datetime, sys
import numpy as np
from mpi4py import MPI

nm = 1e-9
um = 1e-6

Space = build.Space()
#Space.grid = [512,64,4]
#Space.grid = [128,128,400]
Space.grid = [32,32,100]
Space.gridgap = [5*nm, 5*nm, 5*nm]

#Ball = Structure.Sphere(Space,[16,16,110], 10, eps_r=9., mu_r=1., sigma=0.)
#slab = Structure.Box(Space,[0,0,70] , [32,32,100], eps_r=9.23, mu_r=1.2 , sigma=0.)
#slab = Structure.Box(Space,[0,0,100], [32,32,130], eps_r=1.23, mu_r=1.02, sigma=0.)
#slab = Structure.Box(Space,[44,44,80] , [84,84,200], eps_r=9.23, mu_r=1.2 , sigma=0.)
#slab = Structure.Box(Space,[34,34,200], [94,94,300], eps_r=1.23, mu_r=1.02, sigma=0.)
#GaAs = diel.Box([0,0,380], [64,64,460], eps_r=9., mu_r=1., sigma=0.)
#GaAs = diel.Box([0,0,460], [64,64,570], eps_r=5., mu_r=1.12, sigma=0.)

savedir = '/home/ldg/pyscript/hybrid.PSTD.py3/overlap.cpu.oc.2cal/'
UPML = PML.UPML(Space,{'z':'+-'}, npml=10)
#UPML.save_coeff_data(savedir)
Space.Apply_PML(UPML)

Fields = build.Fields(Space)

Src = source.Gaussian(Space)
#Src.wvlen(400*nm,800*nm, .5*nm,spread=0.3)
Src.wvlen = [400*nm, 800*nm, .1*nm, 0.3]

Fields.nsteps = 12001

src_xpos = slice(None,None)
#src_xpos = 256
src_ypos = slice(None,None)
#src_ypos = 256
#src_zpos = slice(None,None)
src_zpos = 30
trs_pos  = -30

Fields.ref_trs_pos = (src_zpos, trs_pos)
#Fields.set_src('Ez',[src_xpos,src_ypos,src_zpos],'soft')
Fields.set_src('Ex',[src_xpos,src_ypos,src_zpos],'soft')
#Fields.set_src('Ey',[src_xpos,src_ypos,src_zpos],'soft')

graphtool = plotting.Graphtool(savedir)

if Space.rank == 0 : 
	t0 = datetime.datetime.now()
	print("Total time step: %d" %(Fields.nsteps))
	print(("Size of a total field array : %05.2f Mbytes" %(Space.Mbytes_of_totalSIZE)))
	print("Simulation start")

for tstep in range(Fields.nsteps):
	
	pulse = Src.pulse(tstep,pick_pos=2200)

	Fields.put_src(pulse)
	Fields.get_ref(tstep)
	Fields.get_trs(tstep)

	Fields.updateHE(tstep, 1./2)
	
	t1 = datetime.datetime.now()
	
	if tstep % 100 == 0:
		Space.comm.Barrier()
		if Space.rank == 0:
			t1 = datetime.datetime.now()
			print(("time: %s, step: %05d, %5.2f%%" %(t1-t0, tstep, 100.*tstep/Fields.nsteps)))
		graphtool.plot2D3D(Fields,'Ex',tstep,xidx=Space.gridxc)
#		graphtool.plot2D3D(Fields,'Ey',tstep,yidx=Space.gridyc, colordeep=2., stride=2, zlim=2.)
#		graphtool.plot2D3D(Fields,'Ez',tstep,zidx=0, colordeep=2, stride=2, zlim=2)
#		graphtool.plot2D3D(Fields,'Hx',tstep,zidx=0, stride=2)
#		graphtool.plot2D3D(Fields,'Hy',tstep,zidx=0, stride=2)

	if tstep == (Fields.nsteps-1):
		Space.comm.Barrier()
		if Space.rank == 0:
			t1 = datetime.datetime.now()
			cal_time = t1 - t0
			print(("time: %s, step: %05d, %5.2f%%" %(cal_time, tstep, 100.*tstep/Fields.nsteps)))

if Space.rank == 0 :
	print("Simulation finished")
	print("Plotting Start")

graphtool.plot_ref_trs(Fields, Src)
graphtool.plot_src(Fields, Src)

if Space.rank == 0 : 

	t2 = datetime.datetime.now()
	print("time: %s" %(t2-t0))
	print("Plotting finished")

	if not os.path.exists("./record") : os.mkdir("./record")
	record_path = "./record/record_%s.txt" %(datetime.date.today())

	if not os.path.exists(record_path):
		f = open( record_path,'a')
		f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n" \
			.format("gridx","gridy","gridz","gridgapx","gridgapy","gridgapz","nsteps","cal_time"))
		f.close()

	f = open( record_path,'a')
	f.write("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n" \
				.format(Space.gridx, Space.gridy, Space.gridz,\
					Space.dx, Space.dy, Space.dz, Fields.nsteps, cal_time))
	f.close()
