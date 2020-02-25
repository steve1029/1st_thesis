import numpy as np
import matplotlib.pyplot as plt
import time, os, datetime, sys
from mpi4py import MPI
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, mu_0, epsilon_0

class Space(object):
	
	def __init__(self, grid, gridgap, dtype, **kwargs):
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

		self.dtype = dtype
		self.comm = MPI.COMM_WORLD
		self.MPIrank = self.comm.Get_rank()
		self.MPIsize = self.comm.Get_size()
		self.hostname = MPI.Get_processor_name()

		assert len(grid)    == 3, "Simulation grid should be a tuple with length 3."
		assert len(gridgap) == 3, "Argument 'gridgap' should be a tuple with length 3."
		
		self.grid = grid
		self.Nx   = self.grid[0]
		self.Ny   = self.grid[1]
		self.Nz   = self.grid[2]
		self.totalSIZE  = self.Nx * self.Ny * self.Nz
		self.Mbytes_of_totalSIZE = (self.dtype(1).nbytes * self.totalSIZE) / 1024 / 1024
		
		self.Nxc = int(self.Nx / 2)
		self.Nyc = int(self.Ny / 2)
		self.Nzc = int(self.Nz / 2)
		
		self.gridgap = gridgap
		self.dx = self.gridgap[0]
		self.dy = self.gridgap[1]
		self.dz = self.gridgap[2]

		self.courant = 1./2

		for key, value in kwargs.items():
			if key == 'courant': self.courant = value

		self.dt = self.courant * min(self.dx, self.dy, self.dz)/c
		self.maxdt = 1. / c / np.sqrt( (1./self.dx)**2 + (1./self.dy)**2 + (1./self.dz)**2 )

		"""For more details about maximum dt,
		see Computational electrodynamics, Allen Taflove, ch 4.7.1"""

		assert self.dt < self.maxdt, "Time interval is too big so that causality is broken. Lower the courant number."
		assert float(self.Nx) % self.MPIsize == 0., "Nx must be a multiple of the number of nodes."
		
		############################################################################
		################# Set the subgrid each node should possess #################
		############################################################################

		xpiece = int(self.Nx/self.MPIsize)
		self.subgrid = [xpiece, self.Ny, self.Nz]

		self.space_eps_onx  = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0
		self.space_eps_ony  = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0
		self.space_eps_onz  = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0

		self.space_eps_offx = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0
		self.space_eps_offy = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0
		self.space_eps_offz = np.ones(self.subgrid, dtype=self.dtype) * epsilon_0

		self.space_mu_onx   = np.ones(self.subgrid, dtype=self.dtype) * mu_0
		self.space_mu_ony   = np.ones(self.subgrid, dtype=self.dtype) * mu_0
		self.space_mu_onz   = np.ones(self.subgrid, dtype=self.dtype) * mu_0

		self.space_mu_offx  = np.ones(self.subgrid, dtype=self.dtype) * mu_0
		self.space_mu_offy  = np.ones(self.subgrid, dtype=self.dtype) * mu_0
		self.space_mu_offz  = np.ones(self.subgrid, dtype=self.dtype) * mu_0

		self.kappa_onx  = np.ones( self.subgrid, dtype=self.dtype)
		self.kappa_ony  = np.ones( self.subgrid, dtype=self.dtype)
		self.kappa_onz  = np.ones( self.subgrid, dtype=self.dtype)

		self.kappa_offx = np.ones( self.subgrid, dtype=self.dtype)
		self.kappa_offy = np.ones( self.subgrid, dtype=self.dtype)
		self.kappa_offz = np.ones( self.subgrid, dtype=self.dtype)

		self.Esigma_onx = np.zeros(self.subgrid, dtype=self.dtype)
		self.Esigma_ony = np.zeros(self.subgrid, dtype=self.dtype)
		self.Esigma_onz = np.zeros(self.subgrid, dtype=self.dtype)

		self.Esigma_offx = np.zeros(self.subgrid, dtype=self.dtype)
		self.Esigma_offy = np.zeros(self.subgrid, dtype=self.dtype)
		self.Esigma_offz = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hsigma_onx = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hsigma_ony = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hsigma_onz = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hsigma_offx = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hsigma_offy = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hsigma_offz = np.zeros(self.subgrid, dtype=self.dtype)

		###############################################################################
		####################### Slices of zgrid that each node got ####################
		###############################################################################
		
		self.xpiece_slices = []
		self.xpiece_indice = []

		for rank in range(1,self.MPIsize):

			xstart = (rank    ) * xpiece
			xend   = (rank + 1) * xpiece

			self.xpiece_slices.append(slice(xstart, xend))
			self.xpiece_indice.append((xstart, xend))

		print("rank: {}, my xindex: {}, my xslice: {}" \
				.format(self.MPIrank, self.xpiece_indice[self.MPIrank], self.xpiece_slices[self.MPIrank]))
			
	def apply_PML(self, region, npml):
	
		#------------------------------------------------------------------------------#
		#---------------------- Specify the PML region in each node -------------------#
		#------------------------------------------------------------------------------#

		myPMLregion_x = None
		myPMLregion_y = None
		myPMLregion_z = None

		for key, value in region.items():

			if   key == 'x':

				if   value == '+':

					if   self.MPIrank == (self.MPIsize-1): myPMLregion_x = '+'

				elif value == '-':

					if   self.MPIrank == 0               : myPMLregion_x = '-'

				elif value == '+-' or '-+':

					if   self.MPIrank == 0               : myPMLregion_x = '-'
					elif self.MPIrank == (self.MPIsize-1): myPMLregion_x = '+'

			elif key == 'y':

				if   value == '+'         : myPMLregion_y = '+'
				elif value == '-'         : myPMLregion_y = '-'
				elif value == '+-' or '-+': myPMLregion_y = '+-'

			elif key == 'z':
	
				if   value == '+'         : myPMLregion_z = '+'
				elif value == '-'         : myPMLregion_z = '-'
				elif value == '+-' or '-+': myPMLregion_z = '+-'

		#----------------------------------------------------------------------------------#
		#----------------------------- Grading of PML region ------------------------------#
		#----------------------------------------------------------------------------------#

		self.npml = npml
		self.PML_applied = True

		rc0      = 1.e-16						# reflection coefficient
		imp_free = np.sqrt(mu_0/epsilon_0)		# impedence in free space
		gO       = 3.							# gradingOrder
		bdw_x    = self.npml * dx				# PML thickness along x (Boundarywidth)
		bdw_y    = self.npml * dy				# PML thickness along y
		bdw_z    = self.npml * dz				# PML thickness along z

		Esigmax_x = -(gO + 1) * np.log(rc0) / (2 * imp_free * bdw_x)
		Esigmax_y = -(gO + 1) * np.log(rc0) / (2 * imp_free * bdw_y)
		Esigmax_z = -(gO + 1) * np.log(rc0) / (2 * imp_free * bdw_z)

		Hsigmax_x = Emaxsig_x * imp_free**2
		Hsigmax_y = Emaxsig_y * imp_free**2
		Hsigmax_z = Emaxsig_z * imp_free**2

		kappamax_x = 5.
		kappamax_y = 5.
		kappamax_z = 5.

		for i in range(self.npml):

			if myPMLregion_x == '-':

				on_grid    = np.float64((self.npml-i   )) / self.npml
				off_grid_m = np.float64((self.npml-i-.5)) / self.npml

				self.Esigma_onx [i,:,:] = Esigmax_x * (on_grid**gO)
				self.Hsigma_onx [i,:,:] = Hsigmax_x * (on_grid**gO)
				self. kappa_onx [i,:,:] = 1 + ((kappamax_x-1) * (on_grid**gO))

				self.Esigma_offx[i,:,:] = Esigmax_x * (off_grid_m**gO)
				self.Hsigma_offx[i,:,:] = Hsigmax_x * (off_grid_m**gO)
				self. kappa_offx[i,:,:] = 1 + ((kappamax_x-1) * (off_grid_m**gO))

			if myPMLregion_x == '+':

				on_grid    = np.float64((self.npml-i   )) / self.npml
				off_grid_p = np.float64((self.npml-i+.5)) / self.npml

				self.Esigma_onx [-i-1,:,:] = Esigmax_x * (on_grid**gO)
				self.Hsigma_onx [-i-1,:,:] = Hsigmax_x * (on_grid**gO)
				self. kappa_onx [-i-1,:,:] = 1 + ((kappamax_x-1) * (on_grid**gO))

				self.Esigma_offx[-i-1,:,:] = Esigmax_x * (off_grid_p**gO)
				self.Hsigma_offx[-i-1,:,:] = Hsigmax_x * (off_grid_p**gO)
				self. kappa_offx[-i-1,:,:] = 1 + ((kappamax_x-1) * (off_grid_p**gO))

			if myPMLregion_x == '+-':

				on_grid    = np.float64((self.npml-i   )) / self.npml
				off_grid_m = np.float64((self.npml-i-.5)) / self.npml
				off_grid_p = np.float64((self.npml-i+.5)) / self.npml

				self.Esigma_onx [i,:,:]    = Esigmax_x * (on_grid**gO)
				self.Hsigma_onx [i,:,:]    = Hsigmax_x * (on_grid**gO)
				self. kappa_onx [i,:,:]    = 1 + ((kappamax_x-1) * (on_grid**gO))

				self.Esigma_offx[i,:,:]    = Esigmax_x * (off_grid_m**gO)
				self.Hsigma_offx[i,:,:]    = Hsigmax_x * (off_grid_m**gO)
				self. kappa_offx[i,:,:]    = 1 + ((kappamax_x-1) * (off_grid_m**gO))

				self.Esigma_onx [-i-1,:,:] = Esigmax_x * (on_grid**gO)
				self.Hsigma_onx [-i-1,:,:] = Hsigmax_x * (on_grid**gO)
				self. kappa_onx [-i-1,:,:] = 1 + ((kappamax_x-1) * (on_grid**gO))

				self.Esigma_offx[-i-1,:,:] = Esigmax_x * (off_grid_p**gO)
				self.Hsigma_offx[-i-1,:,:] = Hsigmax_x * (off_grid_p**gO)
				self. kappa_offx[-i-1,:,:] = 1 + ((kappamax_x-1) * (off_grid_p**gO))

			if myPMLregion_y == '-':

				on_grid    = np.float64((self.npml-j   ))/self.npml
				off_grid_m = np.float64((self.npml-j-.5))/self.npml

				self.Esigma_ony [:,j,:] = Esigmax_y * (on_grid**gO)
				self.Hsigma_ony [:,j,:] = Hsigmax_y * (on_grid**gO)
				self. kappa_ony [:,j,:] = 1 + ((kappamax_y-1) * (on_grid**gO))

				self.Esigma_offy[:,j,:] = Esigmax_y * (off_grid_m**gO)
				self.Hsigma_offy[:,j,:] = Hsigmax_y * (off_grid_m**gO)
				self. kappa_offy[:,j,:] = 1 + ((kappamax_y-1) * (off_grid_m**gO))

			if myPMLregion_y == '+':

				on_grid    = np.float64((self.npml-j   )) / self.npml
				off_grid_p = np.float64((self.npml-j+.5)) / self.npml

				self.Esigma_ony [:,-j-1,:] = Esigmax_y * (on_grid**gO)
				self.Hsigma_ony [:,-j-1,:] = Hsigmax_y * (on_grid**gO)
				self. kappa_ony [:,-j-1,:] = 1 + ((kappamax_y-1) * (on_grid**gO))

				self.Esigma_offy[:,-j-1,:] = Esigmax_y * (off_grid_p**gO)
				self.Hsigma_offy[:,-j-1,:] = Hsigmax_y * (off_grid_p**gO)
				self. kappa_offy[:,-j-1,:] = 1 + ((kappamax_y-1) * (off_grid_p**gO))

			if myPMLregion_y == '+-':

				on_grid    = np.float64((self.npml-j   )) / self.npml
				off_grid_m = np.float64((self.npml-j-.5)) / self.npml
				off_grid_p = np.float64((self.npml-j+.5)) / self.npml

				self.Esigma_ony [:,j,:]    = Esigmax_y * (on_grid**gO)
				self.Hsigma_ony [:,j,:]    = Hsigmax_y * (on_grid**gO)
				self. kappa_ony [:,j,:]    = 1 + ((kappamax_y-1) * (on_grid**gO))

				self.Esigma_offy[:,j,:]    = Esigmax_y * (off_grid_m**gO)
				self.Hsigma_offy[:,j,:]    = Hsigmax_y * (off_grid_m**gO)
				self. kappa_offy[:,j,:]    = 1 + ((kappamax_y-1) * (off_grid_m**gO))

				self.Esigma_ony [:,-j-1,:] = Esigmax_y * (on_grid**gO)
				self.Hsigma_ony [:,-j-1,:] = Hsigmax_y * (on_grid**gO)
				self. kappa_ony [:,-j-1,:] = 1 + ((kappamax_y-1) * (on_grid**gO))

				self.Esigma_offy[:,-j-1,:] = Esigmax_y * (off_grid_p**gO)
				self.Hsigma_offy[:,-j-1,:] = Hsigmax_y * (off_grid_p**gO)
				self. kappa_offy[:,-j-1,:] = 1 + ((kappamax_y-1) * (off_grid_p**gO))


			if myPMLregion_z == '-':

				on_grid    = np.float64((self.npml-k   )) / self.npml
				off_grid_m = np.float64((self.npml-k-.5)) / self.npml

				self.Esigma_onz[:,:,k] = Esigmax_z * (on_grid**gO)
				self.Hsigma_onz[:,:,k] = Hsigmax_z * (on_grid**gO)
				self. kappa_onz[:,:,k] = 1 + ((kappamax_z-1) * (on_grid**gO))

				self.Esigma_offz[:,:,k] = Esigmax_z * (off_grid_m**gO)
				self.Hsigma_offz[:,:,k] = Hsigmax_z * (off_grid_m**gO)
				self. kappa_offz[:,:,k] = 1 + ((kappamax_z-1) * (off_grid_m**gO))

			if myPMLregion_z == '+':

				on_grid    = np.float64((self.npml-k   )) / self.npml
				off_grid_p = np.float64((self.npml-k+.5)) / self.npml

				self.Esigma_onz [:,:,-k-1] = Esigmax_z * (on_grid**gO)
				self.Hsigma_onz [:,:,-k-1] = Hsigmax_z * (on_grid**gO)
				self. kappa_onz [:,:,-k-1] = 1 + ((kappamax_z-1) * (on_grid**gO))

				self.Esigma_offz[:,:,-k-1] = Esigmax_z * (off_grid_p**gO)
				self.Hsigma_offz[:,:,-k-1] = Hsigmax_z * (off_grid_p**gO)
				self. kappa_offz[:,:,-k-1] = 1 + ((kappamax_z-1) * (off_grid_p**gO))

			if myPMLregion_z == '+-':

				on_grid    = np.float64((self.npml-k   )) / self.npml
				off_grid_m = np.float64((self.npml-k-.5)) / self.npml
				off_grid_p = np.float64((self.npml-k+.5)) / self.npml

				self.Esigma_onz[:,:,k] = Esigmax_z * (on_grid**gO)
				self.Hsigma_onz[:,:,k] = Hsigmax_z * (on_grid**gO)
				self. kappa_onz[:,:,k] = 1 + ((kappamax_z-1) * (on_grid**gO))

				self.Esigma_offz[:,:,k] = Esigmax_z * (off_grid_m**gO)
				self.Hsigma_offz[:,:,k] = Hsigmax_z * (off_grid_m**gO)
				self. kappa_offz[:,:,k] = 1 + ((kappamax_z-1) * (off_grid_m**gO))

				self.Esigma_onz [:,:,-k-1] = Esigmax_z * (on_grid**gO)
				self.Hsigma_onz [:,:,-k-1] = Hsigmax_z * (on_grid**gO)
				self. kappa_onz [:,:,-k-1] = 1 + ((kappamax_z-1) * (on_grid**gO))

				self.Esigma_offz[:,:,-k-1] = Esigmax_z * (off_grid_p**gO)
				self.Hsigma_offz[:,:,-k-1] = Hsigmax_z * (off_grid_p**gO)
				self. kappa_offz[:,:,-k-1] = 1 + ((kappamax_z-1) * (off_grid_p**gO))

		print("Rank {0}: PML parameters has been applied along x: {}, y: {}, z: {}" \
					.format(self.MPIrank, myPMLregion_x, myPMLregion_y, myPMLregion_z))

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

		self.myPBCregion_x = None
		self.myPBCregion_y = None
		self.myPBCregion_z = None
		
		for key, value in region.items():

			if   key == 'x':

				if   value == '+':

					if   self.MPIrank == (self.MPIsize-1): myPBCregion_x = '+'

				elif value == '-':

					if   self.MPIrank == 0               : myPBCregion_x = '-'

				elif value == '+-' or '-+':

					if   self.MPIrank == 0               : myPBCregion_x = '-'
					elif self.MPIrank == (self.MPIsize-1): myPBCregion_x = '+'

			elif key == 'y':

				if   value == '+'         : myPMLregion_y = '+'
				elif value == '-'         : myPMLregion_y = '-'
				elif value == '+-' or '-+': myPMLregion_y = '+-'

			elif key == 'z':
	
				if   value == '+'         : myPMLregion_z = '+'
				elif value == '-'         : myPMLregion_z = '-'
				elif value == '+-' or '-+': myPMLregion_z = '+-'


	def space_setting_finished(self):
		"""If structure setting and PML setting is finished, \
		call this method for the 'Space object' to apply your settings.
		"""
	
		self.space_setting_finished = value

		self.Dx_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Dy_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Dz_re = np.zeros(self.subgrid, dtype=self.dtype)

		self.Ex_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ey_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ez_re = np.zeros(self.subgrid, dtype=self.dtype)

		self.Bx_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.By_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Bz_re = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hx_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hy_re = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hz_re = np.zeros(self.subgrid, dtype=self.dtype)

		self.Dx_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Dy_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Dz_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Ex_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ey_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Ez_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Bx_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.By_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Bz_im = np.zeros(self.subgrid, dtype=self.dtype)

		self.Hx_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hy_im = np.zeros(self.subgrid, dtype=self.dtype)
		self.Hz_im = np.zeros(self.subgrid, dtype=self.dtype)

		onEpx = (2 * epsilon_0 * self.kappa_onx) + (self.Esigma_onx * self.dt)
		onEpy = (2 * epsilon_0 * self.kappa_ony) + (self.Esigma_ony * self.dt)
		onEpz = (2 * epsilon_0 * self.kappa_onz) + (self.Esigma_onz * self.dt)
		
		onEmx = (2 * epsilon_0 * self.kappa_onx) - (self.Esigma_onx * self.dt)
		onEmy = (2 * epsilon_0 * self.kappa_ony) - (self.Esigma_ony * self.dt)
		onEmz = (2 * epsilon_0 * self.kappa_onz) - (self.Esigma_onz * self.dt)

		onHpx = (2 * mu_0 * self.kappa_onx) + (self.Hsigma_onx * self.dt)
		onHpy = (2 * mu_0 * self.kappa_ony) + (self.Hsigma_ony * self.dt)
		onHpz = (2 * mu_0 * self.kappa_onz) + (self.Hsigma_onz * self.dt)
		
		onHmx = (2 * mu_0 * self.kappa_onx) - (self.Hsigma_onx * self.dt)
		onHmy = (2 * mu_0 * self.kappa_ony) - (self.Hsigma_ony * self.dt)
		onHmz = (2 * mu_0 * self.kappa_onz) - (self.Hsigma_onz * self.dt)

		offEpx = (2 * epsilon_0 * self.kappa_offx) + (self.Esigma_offx * self.dt)
		offEpy = (2 * epsilon_0 * self.kappa_offy) + (self.Esigma_offy * self.dt)
		offEpz = (2 * epsilon_0 * self.kappa_offz) + (self.Esigma_offz * self.dt)
		
		offEmx = (2 * epsilon_0 * self.kappa_offx) - (self.Esigma_offx * self.dt)
		offEmy = (2 * epsilon_0 * self.kappa_offy) - (self.Esigma_offy * self.dt)
		offEmz = (2 * epsilon_0 * self.kappa_offz) - (self.Esigma_offz * self.dt)

		offHpx = (2 * mu_0 * self.kappa_offx) + (self.Hsigma_offx * self.dt)
		offHpy = (2 * mu_0 * self.kappa_offy) + (self.Hsigma_offy * self.dt)
		offHpz = (2 * mu_0 * self.kappa_offz) + (self.Hsigma_offz * self.dt)
		
		offHmx = (2 * mu_0 * self.kappa_offx) - (self.Hsigma_offx * self.dt)
		offHmy = (2 * mu_0 * self.kappa_offy) - (self.Hsigma_offy * self.dt)
		offHmz = (2 * mu_0 * self.kappa_offz) - (self.Hsigma_offz * self.dt)

		self.CDx1 = (onEmy / onEpy)
		self.CDy1 = (onEmz / onEpz)
		self.CDz1 = (onEmx / onEpx)

		self.CDx2 = 2. * epsilon_0 * self.dt / onEpy
		self.CDy2 = 2. * epsilon_0 * self.dt / onEpz
		self.CDz2 = 2. * epsilon_0 * self.dt / onEpx

		self.CEx1 =  onEmz / onEpz
		self.CEx2 = offEpx / onEpz / self.space_eps_on
		self.CEx3 = offEmx / onEpz / self.space_eps_on * (-1)

		self.CEy1 =  onEmx / onEpx
		self.CEy2 = offEpy / onEpx / self.space_eps_on
		self.CEy3 = offEmy / onEpx / self.space_eps_on * (-1)

		self.CEz1 = onEmy  / onEpy
		self.CEz2 = offEpz / onEpy / self.space_eps_off
		self.CEz3 = offEmz / onEpy / self.space_eps_off * (-1)

		self.CBx1 = offHmy / offHpy
		self.CBy1 = offHmz / offHpz
		self.CBz1 = offHmx / offHpx

		self.CBx2 = 2. * mu_0 * self.dt / offHpy * (-1)
		self.CBy2 = 2. * mu_0 * self.dt / offHpz * (-1)
		self.CBz2 = 2. * mu_0 * self.dt / offHpx * (-1)

		self.CHx1 = offHmz / offHpz
		self.CHx2 =  onHpx / offHpz / self.space_mu_off
		self.CHx3 =  onHmx / offHpz / self.space_mu_off * (-1)

		self.CHy1 = offHmx / offHpx
		self.CHy2 =  onHpy / offHpx / self.space_mu_off
		self.CHy3 =  onHmy / offHpx / self.space_mu_off * (-1)

		self.CHz1 = offHmy / offHpy
		self.CHz2 =  onHpz / offHpy / self.space_mu_on
		self.CHz3 =  onHmz / offHpy / self.space_mu_on * (-1)

	def nsteps(self, nsteps): 
		self._nsteps = nsteps

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

		assert self.nsteps != None, "Set time step first!"

		if ref_pos >= 0: self.ref_pos = ref_pos
		else           : self.ref_pos = ref_pos + self.Nx
		if trs_pos >= 0: self.trs_pos = trs_pos
		else           : self.trs_pos = trs_pos + self.Nx

		##################################################################################
		######################## All rank should know who gets trs #######################
		##################################################################################

		for rank in range(self.MPIsize) : 

			start = self.xpiece_indice[rank][0]
			end   = self.xpiece_indice[rank][1]

			if self.trs_pos >= start and self.trs_pos < end : 
				self.who_get_trs     = rank 
				self.trs_pos_in_node = self.trs_pos - start

		###################################################################################
		####################### All rank should know who gets the ref #####################
		###################################################################################

		for rank in range(self.MPIsize):
			start = self.xpiece_indice[rank][0]
			end   = self.xpiece_indice[rank][1]

			if self.ref_pos >= start and self.ref_pos < end :
				self.who_get_ref     = rank
				self.ref_pos_in_node = self.ref_pos - start 

		#---------------------------------------------------------------------------------#
		#----------------------- Ready to put ref and trs collector ----------------------#
		#---------------------------------------------------------------------------------#

		if   self.MPIrank == self.who_get_trs:
			print("rank %d: I collect trs from %d which is essentially %d in my own grid."\
					 %(self.MPIrank, self.trs_pos, self.trs_pos_in_node))
			self.trs = np.zeros(self.nsteps, dtype=self.dtype) 

		if self.MPIrank == self.who_get_ref: 
			print("rank %d: I collect ref from %d which is essentially %d in my own grid."\
					 %(self.MPIrank, self.ref_pos, self.ref_pos_in_node))
			self.ref = np.zeros(self.nsteps, dtype=self.dtype)

		if self.MPIrank == 0:
			# This arrays are necessary for rank0 to collect src,trs and ref from slave node.
			self.src = np.zeros(self.nsteps, dtype=self.dtype)
			self.trs = np.zeros(self.nsteps, dtype=self.dtype)
			self.ref = np.zeros(self.nsteps, dtype=self.dtype)

		else : pass

	def set_src_pos(self, where, src_start, src_end):
		"""Set the position, type of the source and field.

		PARAMETERS
		----------
		where : string
			The field to put source

			ex)
				'Ex' or 'ex'
				'Ey' or 'ey'
				'Ez' or 'ez'

		position : tuple
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

		self.where       = where
		self.who_put_src = None

		self.src_start  = src_start
		self.src_startx = src_start[0]
		self.src_starty = src_start[1]
		self.src_startz = src_start[2]

		self.src_end  = src_end
		self.src_endx = src_end[0]
		self.src_endy = src_end[1]
		self.src_endz = src_end[2]

		my_startx = self.xpiece_indice[rank][0]
		my_endx   = self.xpiece_indice[rank][1]

		# case 1. x position of source is fixed.
		if self.src_startx == self.src_endx:

			if self.src_startx >= my_startx and self.src_startx <= my_endx:
				self.who_put_src   = self.MPIrank
				self.my_src_startx = self.src_startx - my_startx

				self.src = np.zeros(self.nsteps, dtype=self.dtype)
				print("rank {}, src_startx : {}, my_src_startx: {}"\
						.format(self.MPIrank, self.src_startx, self.my_src_startx))
			else:
				print("rank {}: No putting source".format(self.MPIrank))

		# case 2. x position of source has range.
		elif self.src_startx < self.src_endx:
			raise ValueError("Not developed yet. Sorry.")

		# case 3. x position of source is reversed.
		elif self.src_startx > self.src_endx:
			raise ValueError("src_end[0] is bigger than src_start[0]")

		return None

	def put_src(self, pulse, put_type):

		#------------------------------------------------------------#
		#--------- Put the source into the designated field ---------#
		#------------------------------------------------------------#
		
		self.put_type = put_type
		self.pulse_value = self.dtype(pulse)

		if self.MPIrank == self.who_put_src:

			if   self.put_type == 'soft' :

				if   self.where == 'Ex': self.Ex[x,y,z] += self.pulse_value
				elif self.where == 'Ey': self.Ey[x,y,z] += self.pulse_value
				elif self.where == 'Ez': self.Ez[x,y,z] += self.pulse_value
				elif self.where == 'Hx': self.Hx[x,y,z] += self.pulse_value
				elif self.where == 'Hy': self.Hy[x,y,z] += self.pulse_value
				elif self.where == 'Hz': self.Hz[x,y,z] += self.pulse_value

			elif self.put_type == 'hard' :
	
				if   self.where == 'Ex': self.Ex[x,y,z] = self.pulse_value
				elif self.where == 'Ey': self.Ey[x,y,z] = self.pulse_value
				elif self.where == 'Ez': self.Ez[x,y,z] = self.pulse_value
				elif self.where == 'Hx': self.Hx[x,y,z] = self.pulse_value
				elif self.where == 'Hy': self.Hy[x,y,z] = self.pulse_value
				elif self.where == 'Hz': self.Hz[x,y,z] = self.pulse_value

		else : pass

	def updateH(self,tstep) :
		
		#--------------------------------------------------------------#
		#------------ MPI send Ex and Ey to previous rank -------------#
		#--------------------------------------------------------------#

		if self.MPIrank >= 1 and self.rank <= (self.MPIsize-1):

			send_Ey_first = self.Ey[0,:,:].copy()
			send_Ez_first = self.Ez[0,:,:].copy()

			self.comm.send( send_Ey_first, dest=(self.MPIrank-1), tag=(tstep*100+5))
			self.comm.send( send_Ez_first, dest=(self.MPIrank-1), tag=(tstep*100+6))

		else : pass

		#--------------------------------------------------------------#
		#------------ MPI recv Ex and Ey from next rank ---------------#
		#--------------------------------------------------------------#

		if self.MPIrank >= 0 and self.rank <= (self.MPIsize-2):

			recv_Ey_last = self.comm.recv( source=(self.MPIrank+1), tag=(tstep*100+5))
			recv_Ez_last = self.comm.recv( source=(self.MPIrank+1), tag=(tstep*100+6))

		else : pass

		# First slave node.
		if self.MPIrank == 1:

			xx = self.HxHygrid_per_node[0]
			yy = self.HxHygrid_per_node[1]
			xpiece = self.HxHygrid_per_node[2]

			# Update Hx
			for k in range(xpiece):
				for j in range(yy-1):
					for i in range(xx):

						previous = self.Bx[i,j,k].copy()

						CBx1 = self.CBx1[i,j,k]
						CBx2 = self.CBx2[i,j,k]
						CHx1 = self.CHx1[i,j,k]
						CHx2 = self.CHx2[i,j,k]
						CHx3 = self.CHx3[i,j,k]

						diffzEy = (self.Ey[i,j  ,k+1] - self.Ey[i,j,k]) / self.dz 
						diffyEz = (self.Ez[i,j+1,k  ] - self.Ez[i,j,k]) / self.dy

						self.Bx[i,j,k] = CBx1 * self.Bx[i,j,k] + CBx2 * (diffyEz - diffzEy)
						self.Hx[i,j,k] = CHx1 * self.Hx[i,j,k] + CHx2 * self.Bx[i,j,k] + CHx3 * previous

			# Update Hy
			for k in range(xpiece):
				for j in range(yy):
					for i in range(xx-1):

						previous = self.By[i,j,k].copy()

						CBy1 = self.CBy1[i,j,k]
						CBy2 = self.CBy2[i,j,k]
						CHy1 = self.CHy1[i,j,k]
						CHy2 = self.CHy2[i,j,k]
						CHy3 = self.CHy3[i,j,k]
						
						diffzEx = (self.Ex[i  ,j,k+1] - self.Ex[i,j,k]) / self.dz 
						diffxEz = (self.Ez[i+1,j,k  ] - self.Ez[i,j,k]) / self.dx

						self.By[i,j,k] = CBy1 * self.By[i,j,k] + CBy2 * (diffzEx - diffxEz)
						self.Hy[i,j,k] = CHy1 * self.Hy[i,j,k] + CHy2 * self.By[i,j,k] + CHy3 * previous

			# Update Hz
			xx = self.EzHzgrid_per_node[0]
			yy = self.EzHzgrid_per_node[1]
			xpiece = self.EzHzgrid_per_node[2]

			for k in range(xpiece):
				for j in range(yy-1):
					for i in range(xx-1):

						previous = self.Bz[i,j,k].copy()

						CBz1 = self.CBz1[i,j,k]
						CBz2 = self.CBz2[i,j,k]
						CHz1 = self.CHz1[i,j,k]
						CHz2 = self.CHz2[i,j,k]
						CHz3 = self.CHz3[i,j,k]

						diffxEy = (self.Ey[i+1,j  ,k] - self.Ey[i,j,k]) / self.dx
						diffyEx = (self.Ex[i  ,j+1,k] - self.Ex[i,j,k]) / self.dy
						self.Bz[i,j,k] = CBz1 * self.Bz[i,j,k] + CBz2 * (diffxEy - diffyEx)
						self.Hz[i,j,k] = CHz1 * self.Hz[i,j,k] + CHz2 * self.Bz[i,j,k] + CHz3 * previous

		# Middle slave nodes.
		if self.MPIrank > 1 and self.rank < (self.MPIsize-1):

			xx = self.HxHygrid_per_node[0]
			yy = self.HxHygrid_per_node[1]
			xpiece = self.HxHygrid_per_node[2]

			# Update Hx
			for k in range(1,xpiece):
				for j in range(yy-1):
					for i in range(xx):

						previous = self.Bx[i,j,k].copy()

						CBx1 = self.CBx1[i,j,k]
						CBx2 = self.CBx2[i,j,k]
						CHx1 = self.CHx1[i,j,k]
						CHx2 = self.CHx2[i,j,k]
						CHx3 = self.CHx3[i,j,k]

						diffzEy = (self.Ey[i,j  ,k  ] - self.Ey[i,j,k-1]) / self.dz 
						diffyEz = (self.Ez[i,j+1,k-1] - self.Ez[i,j,k-1]) / self.dy

						self.Bx[i,j,k] = CBx1 * self.Bx[i,j,k] + CBx2 * (diffyEz - diffzEy)
						self.Hx[i,j,k] = CHx1 * self.Hx[i,j,k] + CHx2 * self.Bx[i,j,k] + CHx3 * previous

			# Update Hy
			for k in range(1,xpiece):
				for j in range(yy):
					for i in range(xx-1):

						previous = self.By[i,j,k].copy()

						CBy1 = self.CBy1[i,j,k]
						CBy2 = self.CBy2[i,j,k]
						CHy1 = self.CHy1[i,j,k]
						CHy2 = self.CHy2[i,j,k]
						CHy3 = self.CHy3[i,j,k]
						
						diffzEx = (self.Ex[i  ,j,k  ] - self.Ex[i,j,k-1]) / self.dz 
						diffxEz = (self.Ez[i+1,j,k-1] - self.Ez[i,j,k-1]) / self.dx

						self.By[i,j,k] = CBy1 * self.By[i,j,k] + CBy2 * (diffzEx - diffxEz)
						self.Hy[i,j,k] = CHy1 * self.Hy[i,j,k] + CHy2 * self.By[i,j,k] + CHy3 * previous

			# Update Hz
			xx = self.EzHzgrid_per_node[0]
			yy = self.EzHzgrid_per_node[1]
			xpiece = self.EzHzgrid_per_node[2]

			for k in range(xpiece):
				for j in range(yy-1):
					for i in range(xx-1):

						previous = self.Bz[i,j,k].copy()

						CBz1 = self.CBz1[i,j,k]
						CBz2 = self.CBz2[i,j,k]
						CHz1 = self.CHz1[i,j,k]
						CHz2 = self.CHz2[i,j,k]
						CHz3 = self.CHz3[i,j,k]

						diffxEy = (self.Ey[i+1,j  ,k] - self.Ey[i,j,k]) / self.dx
						diffyEx = (self.Ex[i  ,j+1,k] - self.Ex[i,j,k]) / self.dy
						self.Bz[i,j,k] = CBz1 * self.Bz[i,j,k] + CBz2 * (diffxEy - diffyEx)
						self.Hz[i,j,k] = CHz1 * self.Hz[i,j,k] + CHz2 * self.Bz[i,j,k] + CHz3 * previous

		# Last slave node.
		if self.MPIrank == (self.MPIsize-1):

			xx = self.HxHygrid_per_node[0]
			yy = self.HxHygrid_per_node[1]
			xpiece = self.HxHygrid_per_node[2]

			# Update Hx
			for k in range(1,xpiece-1):
				for j in range(yy-1):
					for i in range(xx):

						previous = self.Bx[i,j,k].copy()

						CBx1 = self.CBx1[i,j,k]
						CBx2 = self.CBx2[i,j,k]
						CHx1 = self.CHx1[i,j,k]
						CHx2 = self.CHx2[i,j,k]
						CHx3 = self.CHx3[i,j,k]

						diffzEy = (self.Ey[i,j  ,k  ] - self.Ey[i,j,k-1]) / self.dz 
						diffyEz = (self.Ez[i,j+1,k-1] - self.Ez[i,j,k-1]) / self.dy

						self.Bx[i,j,k] = CBx1 * self.Bx[i,j,k] + CBx2 * (diffyEz - diffzEy)
						self.Hx[i,j,k] = CHx1 * self.Hx[i,j,k] + CHx2 * self.Bx[i,j,k] + CHx3 * previous

			# Update Hy
			for k in range(1,xpiece-1):
				for j in range(yy):
					for i in range(xx-1):

						previous = self.By[i,j,k].copy()

						CBy1 = self.CBy1[i,j,k]
						CBy2 = self.CBy2[i,j,k]
						CHy1 = self.CHy1[i,j,k]
						CHy2 = self.CHy2[i,j,k]
						CHy3 = self.CHy3[i,j,k]
						
						diffzEx = (self.Ex[i  ,j,k  ] - self.Ex[i,j,k-1]) / self.dz 
						diffxEz = (self.Ez[i+1,j,k-1] - self.Ez[i,j,k-1]) / self.dx

						self.By[i,j,k] = CBy1 * self.By[i,j,k] + CBy2 * (diffzEx - diffxEz)
						self.Hy[i,j,k] = CHy1 * self.Hy[i,j,k] + CHy2 * self.By[i,j,k] + CHy3 * previous

			# Update Hz
			xx = self.EzHzgrid_per_node[0]
			yy = self.EzHzgrid_per_node[1]
			xpiece = self.EzHzgrid_per_node[2]

			for k in range(xpiece):
				for j in range(yy-1):
					for i in range(xx-1):

						previous = self.Bz[i,j,k].copy()

						CBz1 = self.CBz1[i,j,k]
						CBz2 = self.CBz2[i,j,k]
						CHz1 = self.CHz1[i,j,k]
						CHz2 = self.CHz2[i,j,k]
						CHz3 = self.CHz3[i,j,k]

						diffxEy = (self.Ey[i+1,j  ,k] - self.Ey[i,j,k]) / self.dx
						diffyEx = (self.Ex[i  ,j+1,k] - self.Ex[i,j,k]) / self.dy
						self.Bz[i,j,k] = CBz1 * self.Bz[i,j,k] + CBz2 * (diffxEy - diffyEx)
						self.Hz[i,j,k] = CHz1 * self.Hz[i,j,k] + CHz2 * self.Bz[i,j,k] + CHz3 * previous

	def updateE(self, tstep) :

		ft  = np.fft.fftn
		ift = np.fft.ifftn
		nax = np.newaxis

		#---------------------------------------------------------#
		#------------ MPI send Hx and Hy to next rank ------------#
		#---------------------------------------------------------#

		if self.MPIrank > 0 and self.rank < (self.MPIsize-1): # rank 1,2,3,...,n-2

			send_Hx_last = self.Hx[:,:,-1].copy()
			send_Hy_last = self.Hy[:,:,-1].copy()

			self.comm.send(send_Hx_last, dest=(self.MPIrank+1), tag=(tstep*100+1))
			self.comm.send(send_Hy_last, dest=(self.MPIrank+1), tag=(tstep*100+2))

		else : pass

		#---------------------------------------------------------#
		#--------- MPI recv Hx and Hy from previous rank ---------#
		#---------------------------------------------------------#

		if self.MPIrank > 1 and self.rank < self.MPIsize: # rank 2,3,...,n-1

			recv_Hx_first = self.comm.recv( source=(self.MPIrank-1), tag=(tstep*100+1))
			recv_Hy_first = self.comm.recv( source=(self.MPIrank-1), tag=(tstep*100+2) )

			self.Hx[:,:,0] = recv_Hx_first
			self.Hy[:,:,0] = recv_Hy_first
		
		else : pass

		# First slave node.
		if self.MPIrank == 1:

			xx = self.ExEygrid_per_node[0]
			yy = self.ExEygrid_per_node[1]
			xpiece = self.ExEygrid_per_node[2]

			# Update Ex
			for k in range(1,xpiece-1):
				for j in range(1,yy):
					for i in range(xx):

						previous = self.Dx[i,j,k].copy()

						CDx1 = self.CDx1[i,j,k]
						CDx2 = self.CDx2[i,j,k]
						CEx1 = self.CEx1[i,j,k]
						CEx2 = self.CEx2[i,j,k]
						CEx3 = self.CEx3[i,j,k]
						
						diffzHy = (self.Hy[i,j,k] - self.Hy[i,j  ,k-1]) / self.dz 
						diffyHz = (self.Hz[i,j,k] - self.Hz[i,j-1,k  ]) / self.dy

						self.Dx[i,j,k] = CDx1 * self.Dx[i,j,k] + CDx2 * (diffyHz - diffzHy)
						self.Ex[i,j,k] = CEx1 * self.Ex[i,j,k] + CEx2 * self.Dx[i,j,k] + CEx3 * previous

			# Update Ey
			for k in range(1,xpiece-1):
				for j in range(yy):
					for i in range(1,xx):

						previous = self.Dy[i,j,k].copy()

						CDy1 = self.CDy1[i,j,k]
						CDy2 = self.CDy2[i,j,k]
						CEy1 = self.CEy1[i,j,k]
						CEy2 = self.CEy2[i,j,k]
						CEy3 = self.CEy3[i,j,k]

						diffzHx = (self.Hx[i,j,k] - self.Hx[i  ,j,k-1]) / self.dz 
						diffxHz = (self.Hz[i,j,k] - self.Hz[i-1,j,k  ]) / self.dx

						self.Dy[i,j,k] = CDy1 * self.Dy[i,j,k] + CDy2 * (diffzHx - diffxHz)
						self.Ey[i,j,k] = CEy1 * self.Ey[i,j,k] + CEy2 * self.Dy[i,j,k] + CEy3 * previous

			xx = self.EzHzgrid_per_node[0]
			yy = self.EzHzgrid_per_node[1]
			xpiece = self.EzHzgrid_per_node[2]

			# Update Ez
			for k in range(xpiece):
				for j in range(1,yy):
					for i in range(1,xx):

						previous = self.Dz[i,j,k].copy()

						CDz1 = self.CDz1[i,j,k]
						CDz2 = self.CDz2[i,j,k]
						CEz1 = self.CEz1[i,j,k]
						CEz2 = self.CEz2[i,j,k]
						CEz3 = self.CEz3[i,j,k]

						diffxHy = (self.Hy[i,j,k] - self.Hy[i-1,j  ,k]) / self.dx
						diffyHx = (self.Hx[i,j,k] - self.Hx[i  ,j-1,k]) / self.dy
						self.Dz[i,j,k] = CDz1 * self.Dz[i,j,k] + CDz2 * (diffxHy - diffyHx)
						self.Ez[i,j,k] = CEz1 * self.Ez[i,j,k] + CEz2 * self.Dz[i,j,k] + CEz3 * previous

		# Middle slave nodes.
		if self.MPIrank > 1 and self.rank < (self.MPIsize-1):

			xx = self.ExEygrid_per_node[0]
			yy = self.ExEygrid_per_node[1]
			xpiece = self.ExEygrid_per_node[2]

			# Update Ex
			for k in range(xpiece-1):
				for j in range(1,yy):
					for i in range(xx):

						previous = self.Dx[i,j,k].copy()

						CDx1 = self.CDx1[i,j,k]
						CDx2 = self.CDx2[i,j,k]
						CEx1 = self.CEx1[i,j,k]
						CEx2 = self.CEx2[i,j,k]
						CEx3 = self.CEx3[i,j,k]
						
						diffzHy = (self.Hy[i,j,k+1] - self.Hy[i,j  ,k]) / self.dz 
						diffyHz = (self.Hz[i,j,k  ] - self.Hz[i,j-1,k]) / self.dy

						self.Dx[i,j,k] = CDx1 * self.Dx[i,j,k] + CDx2 * (diffyHz - diffzHy)
						self.Ex[i,j,k] = CEx1 * self.Ex[i,j,k] + CEx2 * self.Dx[i,j,k] + CEx3 * previous

			# Update Ey
			for k in range(xpiece-1):
				for j in range(yy):
					for i in range(1,xx):

						previous = self.Dy[i,j,k].copy()

						CDy1 = self.CDy1[i,j,k]
						CDy2 = self.CDy2[i,j,k]
						CEy1 = self.CEy1[i,j,k]
						CEy2 = self.CEy2[i,j,k]
						CEy3 = self.CEy3[i,j,k]

						diffzHx = (self.Hx[i,j,k+1] - self.Hx[i  ,j,k]) / self.dz 
						diffxHz = (self.Hz[i,j,k  ] - self.Hz[i-1,j,k]) / self.dx

						self.Dy[i,j,k] = CDy1 * self.Dy[i,j,k] + CDy2 * (diffzHx - diffxHz)
						self.Ey[i,j,k] = CEy1 * self.Ey[i,j,k] + CEy2 * self.Dy[i,j,k] + CEy3 * previous

			xx = self.EzHzgrid_per_node[0]
			yy = self.EzHzgrid_per_node[1]
			xpiece = self.EzHzgrid_per_node[2]

			# Update Ez
			for k in range(xpiece):
				for j in range(1,yy):
					for i in range(1,xx):

						previous = self.Dz[i,j,k].copy()

						CDz1 = self.CDz1[i,j,k]
						CDz2 = self.CDz2[i,j,k]
						CEz1 = self.CEz1[i,j,k]
						CEz2 = self.CEz2[i,j,k]
						CEz3 = self.CEz3[i,j,k]

						diffxHy = (self.Hy[i,j,k+1] - self.Hy[i-1,j  ,k+1]) / self.dx
						diffyHx = (self.Hx[i,j,k+1] - self.Hx[i  ,j-1,k+1]) / self.dy
						self.Dz[i,j,k] = CDz1 * self.Dz[i,j,k] + CDz2 * (diffxHy - diffyHx)
						self.Ez[i,j,k] = CEz1 * self.Ez[i,j,k] + CEz2 * self.Dz[i,j,k] + CEz3 * previous

		# Last slave node
		if self.MPIrank == (self.MPIsize-1):

			xx = self.ExEygrid_per_node[0]
			yy = self.ExEygrid_per_node[1]
			xpiece = self.ExEygrid_per_node[2]

			# Update Ex
			for k in range(xpiece-1):
				for j in range(1,yy):
					for i in range(xx):

						previous = self.Dx[i,j,k].copy()

						CDx1 = self.CDx1[i,j,k]
						CDx2 = self.CDx2[i,j,k]
						CEx1 = self.CEx1[i,j,k]
						CEx2 = self.CEx2[i,j,k]
						CEx3 = self.CEx3[i,j,k]
						
						diffzHy = (self.Hy[i,j,k+1] - self.Hy[i,j  ,k]) / self.dz 
						diffyHz = (self.Hz[i,j,k  ] - self.Hz[i,j-1,k]) / self.dy

						self.Dx[i,j,k] = CDx1 * self.Dx[i,j,k] + CDx2 * (diffyHz - diffzHy)
						self.Ex[i,j,k] = CEx1 * self.Ex[i,j,k] + CEx2 * self.Dx[i,j,k] + CEx3 * previous

			# Update Ey
			for k in range(xpiece-1):
				for j in range(yy):
					for i in range(1,xx):

						previous = self.Dy[i,j,k].copy()

						CDy1 = self.CDy1[i,j,k]
						CDy2 = self.CDy2[i,j,k]
						CEy1 = self.CEy1[i,j,k]
						CEy2 = self.CEy2[i,j,k]
						CEy3 = self.CEy3[i,j,k]

						diffzHx = (self.Hx[i,j,k+1] - self.Hx[i  ,j,k]) / self.dz 
						diffxHz = (self.Hz[i,j,k  ] - self.Hz[i-1,j,k]) / self.dx

						self.Dy[i,j,k] = CDy1 * self.Dy[i,j,k] + CDy2 * (diffzHx - diffxHz)
						self.Ey[i,j,k] = CEy1 * self.Ey[i,j,k] + CEy2 * self.Dy[i,j,k] + CEy3 * previous

			xx = self.EzHzgrid_per_node[0]
			yy = self.EzHzgrid_per_node[1]
			xpiece = self.EzHzgrid_per_node[2]

			# Update Ez
			for k in range(xpiece):
				for j in range(1,yy):
					for i in range(1,xx):

						previous = self.Dz[i,j,k].copy()

						CDz1 = self.CDz1[i,j,k]
						CDz2 = self.CDz2[i,j,k]
						CEz1 = self.CEz1[i,j,k]
						CEz2 = self.CEz2[i,j,k]
						CEz3 = self.CEz3[i,j,k]

						diffxHy = (self.Hy[i,j,k+1] - self.Hy[i-1,j  ,k+1]) / self.dx
						diffyHx = (self.Hx[i,j,k+1] - self.Hx[i  ,j-1,k+1]) / self.dy
						self.Dz[i,j,k] = CDz1 * self.Dz[i,j,k] + CDz2 * (diffxHy - diffyHx)
						self.Ez[i,j,k] = CEz1 * self.Ez[i,j,k] + CEz2 * self.Dz[i,j,k] + CEz3 * previous

	def get_ref(self,step):

		######################################################################################
		########################## All rank already knows who put src ########################
		######################################################################################

		if self.MPIrank == self.who_get_ref :
			
			if   self.where == 'Ex' : from_the = self.Ex
			elif self.where == 'Ey' : from_the = self.Ey
			elif self.where == 'Ez' : from_the = self.Ez
			elif self.where == 'Hx' : from_the = self.Hx
			elif self.where == 'Hy' : from_the = self.Hy
			elif self.where == 'Hz' : from_the = self.Hz

			self.ref[step] = from_the[:,:,self.ref_pos_in_node].mean() - (self.pulse_value/2./self.courant)
			self.src[step] = self.pulse_value / 2. / self.courant

		else : pass
		
		return None

	def get_trs(self,step) : 
			
		if self.MPIrank == self.who_get_trs :
			
			if   self.where == 'Ex' : from_the = self.Ex
			elif self.where == 'Ey' : from_the = self.Ey
			elif self.where == 'Ez' : from_the = self.Ez
			elif self.where == 'Hx' : from_the = self.Hx
			elif self.where == 'Hy' : from_the = self.Hy
			elif self.where == 'Hz' : from_the = self.Hz

			self.trs[step] = from_the[:,:,self.trs_pos_in_node].mean()

		else : pass

		return None

	def initialize_GPU(processor='cuda'):
		"""Initialize GPU to operate update progress using GPU.
		Here, we basically assume that user has the gpu with cuda processors.
		If one wants to use AMD gpu, set 'processor' argument as 'ocl',
		which is abbreviation of 'OpenCL'.

		PARAMETERS
		----------

		processor : string
			choose processor you want to use. Default is 'cuda'

		RETURNS
		-------

		None"""

		try :
			import reikna.cluda as cld
			print('Start initializeing process for GPU.')
		except ImportError as e :
			print(e)
			print('Reikna is not installed. Plese install reikna by using pip.')
			sys.exit()

		api = cld.get_api(processor)

	#def updateH(self, **kwargs) :
