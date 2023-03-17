import numpy as np
import matplotlib.pyplot as plt
import time, os, datetime, sys, ctypes
from mpi4py import MPI
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, mu_0, epsilon_0

class Space(object):
	
	def __init__(self, grid, gridgap, tsteps, dtype, **kwargs):
		"""Create Simulation Space.

			ex) Space.grid((128,128,600), (50*nm,50*nm,5*nm), dtype=np.float64)

		PARAMETERS
		----------
		grid : tuple
			define the x,y,z grid.

		gridgap : tuple
			define the dx, dy, dz.

		dtype : class numpy dtype
			choose np.float32 or np.float64

		kwargs : string
			
			supported arguments
			-------------------

			courant : float
				Set the courant number. For FDTD, default is 1./2

		RETURNS
		-------
		None
		"""

		self.dtype	  = dtype
		self.comm	  = MPI.COMM_WORLD
		self.MPIrank  = self.comm.Get_rank()
		self.MPIsize  = self.comm.Get_size()
		self.hostname = MPI.Get_processor_name()

		assert len(grid)	== 3, "Simulation grid should be a tuple with length 3."
		assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."

		self.tsteps = tsteps		

		self.grid = grid
		self.Nx   = self.grid[0]
		self.Ny   = self.grid[1]
		self.Nz   = self.grid[2]
		self.totalSIZE	= self.Nx * self.Ny * self.Nz
		self.Mbytes_of_totalSIZE = (self.dtype(1).nbytes * self.totalSIZE) / 1024 / 1024
		
		self.Nxc = int(self.Nx / 2)
		self.Nyc = int(self.Ny / 2)
		self.Nzc = int(self.Nz / 2)
		
		self.gridgap = gridgap
		self.dx = self.gridgap[0]
		self.dy = self.gridgap[1]
		self.dz = self.gridgap[2]

		self.Lx = self.Nx * self.dx
		self.Ly = self.Ny * self.dy
		self.Lz = self.Nz * self.dz

		self.Volume = self.Lx * self.Ly * self.Lz

		if self.MPIrank == 0: print("Volume of the space: {:.2e}" .format(self.Volume))

		self.courant = 1./2

		for key, value in kwargs.items():
			if key == 'courant': self.courant = value

		self.dt = self.courant * min(self.dx, self.dy, self.dz)/c
		self.maxdt = 1. / c / np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )

		assert (c * self.dt * np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )) < 1.

		"""
		For more details about maximum dt,
		see Computational electrodynamics, Allen Taflove, ch 4.7.1
		"""

		self.myPMLregion_x = None
		self.myPMLregion_y = None
		self.myPMLregion_z = None
		self.myPBCregion_x = None
		self.myPBCregion_y = None
		self.myPBCregion_z = None

		assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
		assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
		
		############################################################################
		################# Set the subgrid each node should possess #################
		############################################################################

		self.myNx	 = int(self.Nx/self.MPIsize)
		self.subgrid = [self.myNx, self.Ny, self.Nz]

		self.Ex_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ex_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Ey_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ey_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Ez_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ez_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hx_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hx_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hy_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hy_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hz_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hz_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.eps_Ex = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0
		self.eps_Ey = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0
		self.eps_Ez = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0

		self.mu_Hx	= np.ones(self.subgrid, dtype=self.dtype) * mu_0
		self.mu_Hy	= np.ones(self.subgrid, dtype=self.dtype) * mu_0
		self.mu_Hz	= np.ones(self.subgrid, dtype=self.dtype) * mu_0

		###############################################################################
		####################### Slices of zgrid that each node got ####################
		###############################################################################
		
		self.myNx_slices = []
		self.myNx_indice = []

		for rank in range(self.MPIsize):

			xstart = (rank	  ) * self.myNx
			xend   = (rank + 1) * self.myNx

			self.myNx_slices.append(slice(xstart, xend))
			self.myNx_indice.append((xstart, xend))

		self.comm.Barrier()
		print("rank {:>2}:\tmy xindex: {},\tmy xslice: {}" \
				.format(self.MPIrank, self.myNx_indice[self.MPIrank], self.myNx_slices[self.MPIrank]))

	def apply_PBC(self, region):
		"""Specify the boundary to apply Periodic Boundary Condition.
		Phase compensation is not developed yet.

		PARAMETERS
		----------
		region : dictionary
			ex) {'x':'','y':'+-','z':'+-'}

		RETURNS
		-------
		None
		"""
	
		for key, value in region.items():

			if	 key == 'x':

				if	 value == '+'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '-'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '+-' or value == '-+':

					if	 self.MPIrank == 0				 : self.myPBCregion_x = '-'
					elif self.MPIrank == (self.MPIsize-1): self.myPBCregion_x = '+'

			elif key == 'y':

				if	 value == '+'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '-'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '+-' or value == '-+': self.myPBCregion_y = '+-'

			elif key == 'z':
	
				if	 value == '+'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '-'				   : raise ValueError("input '+-' or '-+'.")
				elif value == '+-' or value == '-+': self.myPBCregion_z = '+-'

		self.comm.Barrier()
		print("PBC region of rank: {}, x: {}, y: {}, z: {}" \
				.format(self.MPIrank, self.myPBCregion_x, self.myPBCregion_y, self.myPBCregion_z))

	def apply_structures(self, structure_list):
		self.structure_list = structure_list

	def apply_PML(self, region, npml):
		self.PMLregion = region

	def set_ref_trs_pos(self, ref_pos, trs_pos):
		"""Set x position to collect srcref and trs

		PARAMETERS
		----------
		pos : tuple
				x index of ref position and trs position

		RETURNS
		-------
		None
		"""

		assert self.tsteps != None, "Set time tstep first!"

		if ref_pos >= 0: self.ref_pos = ref_pos
		else		   : self.ref_pos = ref_pos + self.Nx
		if trs_pos >= 0: self.trs_pos = trs_pos
		else		   : self.trs_pos = trs_pos + self.Nx

		##################################################################################
		######################## All rank should know who gets trs #######################
		##################################################################################

		for rank in range(self.MPIsize) : 

			start = self.myNx_indice[rank][0]
			end   = self.myNx_indice[rank][1]

			if self.trs_pos >= start and self.trs_pos < end : 
				self.who_get_trs	 = rank 
				self.trs_pos_in_node = self.trs_pos - start

		###################################################################################
		####################### All rank should know who gets the ref #####################
		###################################################################################

		for rank in range(self.MPIsize):
			start = self.myNx_indice[rank][0]
			end   = self.myNx_indice[rank][1]

			if self.ref_pos >= start and self.ref_pos < end :
				self.who_get_ref	 = rank
				self.ref_pos_in_node = self.ref_pos - start 

		#---------------------------------------------------------------------------------#
		#----------------------- Ready to put ref and trs collector ----------------------#
		#---------------------------------------------------------------------------------#

		self.comm.Barrier()
		if	 self.MPIrank == self.who_get_trs:
			print("rank %d: I collect trs from %d which is essentially %d in my own grid."\
					 %(self.MPIrank, self.trs_pos, self.trs_pos_in_node))
			self.trs = np.zeros(self.tsteps, dtype=self.dtype) 

		if self.MPIrank == self.who_get_ref: 
			print("rank %d: I collect ref from %d which is essentially %d in my own grid."\
					 %(self.MPIrank, self.ref_pos, self.ref_pos_in_node))
			self.ref = np.zeros(self.tsteps, dtype=self.dtype)

	def set_src_pos(self, src_start, src_end):
		"""Set the position, type of the source and field.

		PARAMETERS
		----------
		src_start : tuple
		src_end   : tuple

			A tuple which has three ints as its elements.
			The elements defines the position of the source in the field.
			
			ex)
				1. point source
					src_start: (30, 30, 30), src_end: (30, 30, 30)
				2. line source
					src_start: (30, 30, 0), src_end: (30, 30, Space.Nz)
				3. plane wave
					src_start: (30,0,0), src_end: (30, Space.Ny, Space.Nz)

		RETURNS
		-------
		None
		"""

		assert len(src_start) == 3, "src_start argument is a list or tuple with length 3."
		assert len(src_end)   == 3, "src_end argument is a list or tuple with length 3."

		self.who_put_src = None

		self.src_start	= src_start
		self.src_startx = src_start[0]
		self.src_starty = src_start[1]
		self.src_startz = src_start[2]

		self.src_end  = src_end
		self.src_endx = src_end[0]
		self.src_endy = src_end[1]
		self.src_endz = src_end[2]

		#----------------------------------------------------------------------#
		#--------- All rank should know who put src to plot src graph ---------#
		#----------------------------------------------------------------------#

		self.comm.Barrier()
		for rank in range(self.MPIsize):

			my_startx = self.myNx_indice[rank][0]
			my_endx   = self.myNx_indice[rank][1]

			# case 1. x position of source is fixed.
			if self.src_startx == (self.src_endx-1):

				if self.src_startx >= my_startx and self.src_endx < my_endx:
					self.who_put_src   = rank

					if self.MPIrank == self.who_put_src:
						self.my_src_startx = self.src_startx - my_startx
						self.my_src_endx   = self.src_endx	 - my_startx

						self.src_re = np.zeros(self.tsteps, dtype=self.dtype)
						self.src_im = np.zeros(self.tsteps, dtype=self.dtype)

						print("rank {:>2}: src_startx : {}, my_src_startx: {}, my_src_endx: {}"\
								.format(self.MPIrank, self.src_startx, self.my_src_startx, self.my_src_endx))
					#else:
					#	print("rank {:>2}: I don't put source".format(self.MPIrank))

				else: continue

			# case 2. x position of source has range.
			elif self.src_startx < self.src_endx:
				raise ValueError("Not developed yet. Sorry.")

			# case 3. x position of source is reversed.
			elif self.src_startx > self.src_endx:
				raise ValueError("src_end[0] is bigger than src_start[0]")

			else:
				raise IndexError("x position of src is not defined!")

	def put_src(self, where_re, where_im, pulse_re, pulse_im, put_type):
		"""Put source at the designated postion set by set_src_pos method.
		
		PARAMETERS
		----------	
		where : string
			ex)
				'Ex_re' or 'ex_re'
				'Ey_re' or 'ey_re'
				'Ez_re' or 'ez_re'

				'Ex_im' or 'ex_im'
				'Ey_im' or 'ey_im'
				'Ez_im' or 'ez_im'

		pulse : float
			float returned by source.pulse_re or source.pulse_im.

		put_type : string
			'soft' or 'hard'

		"""
		#------------------------------------------------------------#
		#--------- Put the source into the designated field ---------#
		#------------------------------------------------------------#

		self.put_type = put_type

		self.where_re = where_re
		self.where_im = where_im
		
		self.pulse_re = self.dtype(pulse_re)
		self.pulse_im = self.dtype(pulse_im)

		if self.MPIrank == self.who_put_src:

			x = slice(self.my_src_startx, self.my_src_endx)
			y = slice(self.src_starty	, self.src_endy   )
			z = slice(self.src_startz	, self.src_endz   )

			if	 self.put_type == 'soft':

				if (self.where_re == 'Ex_re') or (self.where_re == 'ex_re'): self.Ex_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Ey_re') or (self.where_re == 'ey_re'): self.Ey_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Ez_re') or (self.where_re == 'ez_re'): self.Ez_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Hx_re') or (self.where_re == 'hx_re'): self.Hx_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Hy_re') or (self.where_re == 'hy_re'): self.Hy_re[x,y,z] += self.pulse_re
				if (self.where_re == 'Hz_re') or (self.where_re == 'hz_re'): self.Hz_re[x,y,z] += self.pulse_re

				if (self.where_im == 'Ex_im') or (self.where_im == 'ex_im'): self.Ex_im[x,y,z] += self.pulse_im
				if (self.where_im == 'Ey_im') or (self.where_im == 'ey_im'): self.Ey_im[x,y,z] += self.pulse_im
				if (self.where_im == 'Ez_im') or (self.where_im == 'ez_im'): self.Ez_im[x,y,z] += self.pulse_im
				if (self.where_im == 'Hx_im') or (self.where_im == 'hx_im'): self.Hx_im[x,y,z] += self.pulse_im
				if (self.where_im == 'Hy_im') or (self.where_im == 'hy_im'): self.Hy_im[x,y,z] += self.pulse_im
				if (self.where_im == 'Hz_im') or (self.where_im == 'hz_im'): self.Hz_im[x,y,z] += self.pulse_im

			elif self.put_type == 'hard':
	
				if (self.where_re == 'Ex_re') or (self.where_re == 'ex_re'): self.Ex_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Ey_re') or (self.where_re == 'ey_re'): self.Ey_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Ez_re') or (self.where_re == 'ez_re'): self.Ez_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Hx_re') or (self.where_re == 'hx_re'): self.Hx_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Hy_re') or (self.where_re == 'hy_re'): self.Hy_re[x,y,z] = self.pulse_re
				if (self.where_re == 'Hz_re') or (self.where_re == 'hz_re'): self.Hz_re[x,y,z] = self.pulse_re

				if (self.where_im == 'Ex_im') or (self.where_im == 'ex_im'): self.Ex_im[x,y,z] = self.pulse_im
				if (self.where_im == 'Ey_im') or (self.where_im == 'ey_im'): self.Ey_im[x,y,z] = self.pulse_im
				if (self.where_im == 'Ez_im') or (self.where_im == 'ez_im'): self.Ez_im[x,y,z] = self.pulse_im
				if (self.where_im == 'Hx_im') or (self.where_im == 'hx_im'): self.Hx_im[x,y,z] = self.pulse_im
				if (self.where_im == 'Hy_im') or (self.where_im == 'hy_im'): self.Hy_im[x,y,z] = self.pulse_im
				if (self.where_im == 'Hz_im') or (self.where_im == 'hz_im'): self.Hz_im[x,y,z] = self.pulse_im

			else:
				raise ValueError("Please insert 'soft' or 'hard'")

	def init_update_equations(self, core_omp, PBC_omp):
		"""Setter for PML, structures

			After applying structures, PML are finished, call this method.
			It will prepare DLL for update equations.
		"""

		ptr1d = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=1, flags='C_CONTIGUOUS')
		ptr2d = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=2, flags='C_CONTIGUOUS')
		ptr3d = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=3, flags='C_CONTIGUOUS')

		# Call core update equations

		self.core_omp = core_omp
		self.PBC_omp  = PBC_omp

		if   self.core_omp == False:
			self.clib_updateEH = ctypes.cdll.LoadLibrary("./core.so")
			#print("rank:{}, core.so has been loaded." .format(self.MPIrank))
		elif self.core_omp == True:
			self.clib_updateEH = ctypes.cdll.LoadLibrary("./core.omp.so")
			#print("rank:{}, core.omp.so has been loaded." .format(self.MPIrank))
		else: raise ValueError

		self.clib_updateEH.update_inner_E.restype	 = None
		self.clib_updateEH.update_inner_H.restype	 = None
		self.clib_updateEH.update_last_HyHz.restype  = None
		self.clib_updateEH.update_first_EyEz.restype = None

		self.clib_updateEH.update_inner_E.argtypes	  = [	ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr3d, ptr3d, ptr3d, \
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_updateEH.update_inner_H.argtypes	  = [	ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr3d, ptr3d, ptr3d, \
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_updateEH.update_first_EyEz.argtypes = [	ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr2d, ptr2d, ptr2d, ptr2d, ptr3d, ptr3d, \
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_updateEH.update_last_HyHz.argtypes  = [	ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, \
															ptr2d, ptr2d, ptr2d, ptr2d, ptr3d, ptr3d, \
															ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
															ctypes.c_int, ctypes.c_int, ctypes.c_int]

		# Call PBC update equations
		if   self.PBC_omp == False:
			self.clib_PBC = ctypes.cdll.LoadLibrary("./PBC.so")
			#print("rank:{}, PBC.so has been loaded." .format(self.MPIrank))
		elif self.PBC_omp == True :
			self.clib_PBC = ctypes.cdll.LoadLibrary("./PBC.omp.so")
			#print("rank:{}, PBC.omp.so has been loaded." .format(self.MPIrank))
		else: raise ValueError

		self.clib_PBC.apply_PBC_inner_yminus.restype = None
		self.clib_PBC.apply_PBC_outer_yminus.restype = None

		self.clib_PBC.apply_PBC_inner_yplus.restype  = None
		self.clib_PBC.apply_PBC_outer_yplus.restype  = None

		self.clib_PBC.apply_PBC_inner_zminus.restype = None
		self.clib_PBC.apply_PBC_outer_zminus.restype = None

		self.clib_PBC.apply_PBC_inner_zplus.restype  = None
		self.clib_PBC.apply_PBC_outer_zplus.restype  = None

		self.clib_PBC.apply_PBC_inner_yminus.argtypes = [						\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d,							\
			ptr3d, ptr3d,														\
			ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_outer_yminus.argtypes = [						\
			ptr3d, ptr3d,														\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr2d, ptr2d,														\
			ptr3d,																\
			ctypes.c_double, ctypes.c_double, ctypes.c_double,					\
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_inner_yplus.argtypes = [						\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d,							\
			ptr3d, ptr3d,														\
			ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_outer_yplus.argtypes = [						\
			ptr3d, ptr3d,														\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr2d, ptr2d,														\
			ptr3d,																\
			ctypes.c_double, ctypes.c_double, ctypes.c_double,					\
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_inner_zminus.argtypes = [						\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d,							\
			ptr3d, ptr3d,														\
			ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_outer_zminus.argtypes = [						\
			ptr3d, ptr3d,														\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr2d, ptr2d,														\
			ptr3d,																\
			ctypes.c_double, ctypes.c_double, ctypes.c_double,					\
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_inner_zplus.argtypes = [						\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr3d, ptr3d, ptr3d, ptr3d, ptr3d, ptr3d,							\
			ptr3d, ptr3d,														\
			ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

		self.clib_PBC.apply_PBC_outer_zplus.argtypes = [						\
			ptr3d, ptr3d,														\
			ptr3d, ptr3d, ptr3d, ptr3d,											\
			ptr2d, ptr2d,														\
			ptr3d,																\
			ctypes.c_double, ctypes.c_double, ctypes.c_double,					\
			ctypes.c_int, ctypes.c_int, ctypes.c_int]

	def updateH(self,tstep) :
		
		#--------------------------------------------------------------#
		#------------ MPI send Ex and Ey to previous rank -------------#
		#--------------------------------------------------------------#

		if (self.MPIrank > 0) and (self.MPIrank < self.MPIsize):

			sendEyfirst_re = self.Ey_re[0,:,:].copy()
			sendEyfirst_im = self.Ey_im[0,:,:].copy()
			sendEzfirst_re = self.Ez_re[0,:,:].copy()
			sendEzfirst_im = self.Ez_im[0,:,:].copy()

			self.comm.send( sendEyfirst_re, dest=(self.MPIrank-1), tag=(tstep*100+9 ))
			self.comm.send( sendEyfirst_im, dest=(self.MPIrank-1), tag=(tstep*100+10))
			self.comm.send( sendEzfirst_re, dest=(self.MPIrank-1), tag=(tstep*100+11))
			self.comm.send( sendEzfirst_im, dest=(self.MPIrank-1), tag=(tstep*100+12))

		else: pass

		#-----------------------------------------------------------#
		#------------ MPI recv Ex and Ey from next rank ------------#
		#-----------------------------------------------------------#

		if (self.MPIrank > (-1)) and (self.MPIrank < (self.MPIsize-1)):

			recvEylast_re = self.comm.recv( source=(self.MPIrank+1), tag=(tstep*100+9 ))
			recvEylast_im = self.comm.recv( source=(self.MPIrank+1), tag=(tstep*100+10))
			recvEzlast_re = self.comm.recv( source=(self.MPIrank+1), tag=(tstep*100+11))
			recvEzlast_im = self.comm.recv( source=(self.MPIrank+1), tag=(tstep*100+12))
		else: pass

		# All nodes update their inner Hx, Hy and Hz.
		self.clib_updateEH.update_inner_H(	self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, self.Hz_re, self.Hz_im, \
											self.Ex_re, self.Ex_im, self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
											self.mu_Hx, self.mu_Hy, self.mu_Hz, \
											self.dt, self.dx, self.dy, self.dz, self.myNx, self.Ny, self.Nz)

		# First node and Middle nodes update their Hy and Hz at x=myNx-1.
		if self.MPIrank > (-1) and self.MPIrank < (self.MPIsize-1):

			self.clib_updateEH.update_last_HyHz(self.Hy_re, self.Hy_im, self.Hz_re, self.Hz_im, \
												self.Ex_re, self.Ex_im, self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
												recvEylast_re, recvEylast_im, recvEzlast_re, recvEzlast_im, \
												self.mu_Hy, self.mu_Hz, \
												self.dt, self.dx, self.dy, self.dz, \
												self.myNx, self.Ny, self.Nz)	
		else: pass

		# Apply PBC along y when it is given.
		if self.myPBCregion_y == '+-':

			self.clib_PBC.apply_PBC_inner_yplus(self.Hx_re, self.Hx_im, self.Hz_re, self.Hz_im, \
												self.Ex_re, self.Ex_im, self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
												self.mu_Hx, self.mu_Hz, \
												self.dt, self.dx, self.dy, self.dz, \
												self.myNx, self.Ny, self.Nz)

			if (self.MPIrank >= 0) and (self.MPIrank <= (self.MPIsize-2)):

				self.clib_PBC.apply_PBC_outer_yplus(self.Hz_re, self.Hz_im,	\
													self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
													recvEylast_re, recvEylast_im, \
													self.mu_Hz, \
													self.dt, self.dx, self.dy, \
													self.myNx, self.Ny, self.Nz)
			else: pass
		else: pass

		# Apply PBC along z when it is given.
		if self.myPBCregion_z == '+-':

			self.clib_PBC.apply_PBC_inner_zplus(self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, \
												self.Ex_re, self.Ex_im, self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
												self.mu_Hx, self.mu_Hy, \
												self.dt, self.dx, self.dy, self.dz, \
												self.myNx, self.Ny, self.Nz)

			if (self.MPIrank >= 0) and (self.MPIrank <= (self.MPIsize-2)):

				self.clib_PBC.apply_PBC_outer_zplus(self.Hy_re, self.Hy_im,	\
													self.Ex_re, self.Ex_im, self.Ez_re, self.Ez_im, \
													recvEzlast_re, recvEzlast_im, \
													self.mu_Hy, \
													self.dt, self.dx, self.dy, \
													self.myNx, self.Ny, self.Nz)
			else: pass
		else: pass

	def updateE(self, tstep):

		#---------------------------------------------------------#
		#------------ MPI send Hy and Hz to next rank ------------#
		#---------------------------------------------------------#

		if self.MPIrank > (-1) and self.MPIrank < (self.MPIsize-1):

			sendHylast_re = self.Hy_re[-1,:,:].copy()
			sendHylast_im = self.Hy_im[-1,:,:].copy()
			sendHzlast_re = self.Hz_re[-1,:,:].copy()
			sendHzlast_im = self.Hz_im[-1,:,:].copy()

			self.comm.send(sendHylast_re, dest=(self.MPIrank+1), tag=(tstep*100+3))
			self.comm.send(sendHylast_im, dest=(self.MPIrank+1), tag=(tstep*100+4))
			self.comm.send(sendHzlast_re, dest=(self.MPIrank+1), tag=(tstep*100+5))
			self.comm.send(sendHzlast_im, dest=(self.MPIrank+1), tag=(tstep*100+6))
		
		else: pass

		#---------------------------------------------------------#
		#--------- MPI recv Hy and Hz from previous rank ---------#
		#---------------------------------------------------------#

		if self.MPIrank > 0 and self.MPIrank < self.MPIsize:

			recvHyfirst_re = self.comm.recv( source=(self.MPIrank-1), tag=(tstep*100+3))
			recvHyfirst_im = self.comm.recv( source=(self.MPIrank-1), tag=(tstep*100+4))
			recvHzfirst_re = self.comm.recv( source=(self.MPIrank-1), tag=(tstep*100+5))
			recvHzfirst_im = self.comm.recv( source=(self.MPIrank-1), tag=(tstep*100+6))
		
		else: pass

		# All nodes update their inner Ex, Ey and Ez.
		self.clib_updateEH.update_inner_E(	self.Ex_re, self.Ex_im, self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
											self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, self.Hz_re, self.Hz_im, \
											self.eps_Ex, self.eps_Ey, self.eps_Ez, \
											self.dt, self.dx, self.dy, self.dz, self.myNx, self.Ny, self.Nz)

		# Middle nodes and the last node update their Ey and Ez at x=0.
		if self.MPIrank > 0 and self.MPIrank < self.MPIsize:

			self.clib_updateEH.update_first_EyEz(	self.Ey_re, self.Ey_im, self.Ez_re, self.Ez_im, \
													self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, self.Hz_re, self.Hz_im, \
													recvHyfirst_re, recvHyfirst_im, recvHzfirst_re, recvHzfirst_im, \
													self.eps_Ey, self.eps_Ez, \
													self.dt, self.dx, self.dy, self.dz, \
													self.myNx, self.Ny, self.Nz)

		else: pass

		# Apply PBC along y when it is given.
		if self.myPBCregion_y == '+-':

			self.clib_PBC.apply_PBC_inner_yminus(	self.Ex_re, self.Ex_im, self.Ez_re, self.Ez_im, \
													self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, self.Hz_re, self.Hz_im, \
													self.eps_Ex, self.eps_Ez, \
													self.dt, self.dx, self.dy, self.dz, \
													self.myNx, self.Ny, self.Nz)

			if (self.MPIrank > 0) and (self.MPIrank < self.MPIsize):

				self.clib_PBC.apply_PBC_outer_yminus(	self.Ez_re, self.Ez_im,	\
														self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, \
														recvHyfirst_re, recvHyfirst_im, \
														self.eps_Ez, \
														self.dt, self.dx, self.dy, \
														self.myNx, self.Ny, self.Nz)
			else: pass
		else: pass

		# Apply PBC along z when it is given.
		if self.myPBCregion_z == '+-':

			self.clib_PBC.apply_PBC_inner_zminus(	self.Ex_re, self.Ex_im, self.Ey_re, self.Ey_im, \
													self.Hx_re, self.Hx_im, self.Hy_re, self.Hy_im, self.Hz_re, self.Hz_im, \
													self.eps_Ex, self.eps_Ey, \
													self.dt, self.dx, self.dy, self.dz, \
													self.myNx, self.Ny, self.Nz)

			if (self.MPIrank >= 1) and (self.MPIrank <= (self.MPIsize-1)):

				self.clib_PBC.apply_PBC_outer_zminus(	self.Ey_re, self.Ey_im,	\
														self.Hx_re, self.Hx_im, self.Hz_re, self.Hz_im, \
														recvHzfirst_re, recvHzfirst_im, \
														self.eps_Ey, \
														self.dt, self.dx, self.dy, \
														self.myNx, self.Ny, self.Nz)
			else: pass
		else: pass

	def get_src(self,tstep):

		if self.MPIrank == self.who_put_src:
			
			if	 self.where_re == 'Ex_re': from_the = self.Ex_re
			elif self.where_re == 'Ey_re': from_the = self.Ey_re
			elif self.where_re == 'Ez_re': from_the = self.Ez_re
			elif self.where_re == 'Hx_re': from_the = self.Hx_re
			elif self.where_re == 'Hy_re': from_the = self.Hy_re
			elif self.where_re == 'Hz_re': from_the = self.Hz_re

			elif self.where_im == 'Ex_im': from_the = self.Ex_im
			elif self.where_im == 'Ey_im': from_the = self.Ey_im
			elif self.where_im == 'Ez_im': from_the = self.Ez_im
			elif self.where_im == 'Hx_im': from_the = self.Hx_im
			elif self.where_im == 'Hy_im': from_the = self.Hy_im
			elif self.where_im == 'Hz_im': from_the = self.Hz_im

			if self.pulse_re != None: self.src_re[tstep] = self.pulse_re / 2. / self.courant
			if self.pulse_im != None: self.src_im[tstep] = self.pulse_im / 2. / self.courant

	def get_ref(self,tstep):

		######################################################################################
		########################## All rank already knows who put src ########################
		######################################################################################

		if self.MPIrank == self.who_get_ref:
			
			if	 self.where == 'Ex': from_the = self.Ex
			elif self.where == 'Ey': from_the = self.Ey
			elif self.where == 'Ez': from_the = self.Ez
			elif self.where == 'Hx': from_the = self.Hx
			elif self.where == 'Hy': from_the = self.Hy
			elif self.where == 'Hz': from_the = self.Hz

			self.ref[tstep] = from_the[:,:,self.ref_pos_in_node].mean() - (self.pulse_value/2./self.courant)

		else : pass
		
	def get_trs(self,tstep) : 
			
		if self.MPIrank == self.who_get_trs:
			
			if	 self.where == 'Ex': from_the = self.Ex
			elif self.where == 'Ey': from_the = self.Ey
			elif self.where == 'Ez': from_the = self.Ez
			elif self.where == 'Hx': from_the = self.Hx
			elif self.where == 'Hy': from_the = self.Hy
			elif self.where == 'Hz': from_the = self.Hz

			self.trs[tstep] = from_the[:,:,self.trs_pos_in_node].mean()

		else : pass
