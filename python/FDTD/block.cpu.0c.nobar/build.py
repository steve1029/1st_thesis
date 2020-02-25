import numpy as np
import matplotlib.pyplot as plt
import time, os, datetime, sys
from mpi4py import MPI
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, mu_0, epsilon_0

class Space(object):
	
	def __init__(self,**kwargs):

		self.dimension = 3
		self.dtype = np.float32
		
		self.comm = MPI.COMM_WORLD
		self.rank = self.comm.Get_rank()
		self.size = self.comm.Get_size()
		self.hostname = MPI.Get_processor_name()

		assert self.size > 1, "Only one computer in MPI_CONTEXT."

		for key, value in list(kwargs.items()):
			if key == 'dtype':
				self.dtype = value
				if self.rank == 0: print("Computation data type has been set to ", value)
	
	@property	
	def grid(self): return self._grid
	
	@grid.setter
	def grid(self,grid):
		
		assert len(grid) == 3, "Simulation grid should be the 3 dimensional array."
		
		self._grid = grid
		self.gridx = self._grid[0]
		self.gridy = self._grid[1]
		self.gridz = self._grid[2]
		self.totalSIZE  = self.gridx * self.gridy * self.gridz
		self.Mbytes_of_totalSIZE = (self.dtype(1).nbytes * self.totalSIZE) / 1024 / 1024
		
		self.gridxc = int(self.gridx / 2)
		self.gridyc = int(self.gridy / 2)
		self.gridzc = int(self.gridz / 2)
		
		assert (float(self.gridz) % (self.size-1)) == 0., "z-grid must be a multiple of the number of slave nodes"
		
		zz = int(self.gridz/(self.size-1))

		############################################################################
		################# Set the subgrid each node should possess #################
		############################################################################

		# All the first, middle and last slave nodes have two kinds of grids
		# such that one for Ex,Ey,Hz and the other for Hx,Hy,Ez.
		# I'll call the grid for Ex,Ey and Hz as 'EEH grid' and
		# the other grid, which is for Hx,Hy and Ez, as 'HHE grid'.
		# Also, all nodes must know their own grid which does not have a 'fake grid' for MPI communication.

		if self.rank == 1:
			self.OWN_grid_per_node = [self.gridx, self.gridy, zz  ]	# Original grid of each node.
			self.EzHzgrid_per_node = [self.gridx, self.gridy, zz  ]	# Grid for Ez, Hz
			self.ExEygrid_per_node = [self.gridx, self.gridy, zz+1]	# Grid for Ex, Ey
			self.HxHygrid_per_node = [self.gridx, self.gridy, zz  ]	# Grid for Hx, Hy

			print("rank: %d, name: %s, has OWN grid %s" %(self.rank, self.hostname, tuple(self.OWN_grid_per_node)))

		if self.rank > 1 and self.rank < (self.size-1):
			self.OWN_grid_per_node = [self.gridx, self.gridy, zz  ]	# Original grid of each node.
			self.EzHzgrid_per_node = [self.gridx, self.gridy, zz  ]	# Grid for Ez, Hz
			self.ExEygrid_per_node = [self.gridx, self.gridy, zz+1]	# Grid for Ex, Ey
			self.HxHygrid_per_node = [self.gridx, self.gridy, zz+1]	# Grid for Hx, Hy

			print("rank: %d, name: %s, has OWN grid %s" %(self.rank, self.hostname, tuple(self.OWN_grid_per_node)))

		if self.rank == (self.size-1):
			self.OWN_grid_per_node = [self.gridx, self.gridy, zz  ]	# Original grid of each node.
			self.EzHzgrid_per_node = [self.gridx, self.gridy, zz  ]	# Grid for Ez, Hz
			self.ExEygrid_per_node = [self.gridx, self.gridy, zz  ]	# Grid for Ex, Ey
			self.HxHygrid_per_node = [self.gridx, self.gridy, zz+1]	# Grid for Hx, Hy

			print("rank: %d, name: %s, has OWN grid %s" %(self.rank, self.hostname, tuple(self.OWN_grid_per_node)))

		if self.rank == 0 :

			if   self.dtype == np.complex128: coffdtype = np.float64
			elif self.dtype == np.complex64 : coffdtype = np.float32
			elif self.dtype == np.float64   : coffdtype = np.float64
			elif self.dtype == np.float32   : coffdtype = np.float32
			else:
				raise TypeError

			print("Coefficient array dtype has been set to ", coffdtype)

			self.space_eps_on  = np.ones(self.grid, dtype=coffdtype) * epsilon_0
			self.space_eps_off = np.ones(self.grid, dtype=coffdtype) * epsilon_0
			self.space_mu_on   = np.ones(self.grid, dtype=coffdtype) * mu_0
			self.space_mu_off  = np.ones(self.grid, dtype=coffdtype) * mu_0

			#----------------------------------------------------#
			self.kappa_onx  = np.ones( self.grid, dtype=coffdtype)
			self.kappa_ony  = np.ones( self.grid, dtype=coffdtype)
			self.kappa_onz  = np.ones( self.grid, dtype=coffdtype)

			self.kappa_offx = np.ones( self.grid, dtype=coffdtype)
			self.kappa_offy = np.ones( self.grid, dtype=coffdtype)
			self.kappa_offz = np.ones( self.grid, dtype=coffdtype)

			self.Esigma_onx = np.zeros(self.grid, dtype=coffdtype)
			self.Esigma_ony = np.zeros(self.grid, dtype=coffdtype)
			self.Esigma_onz = np.zeros(self.grid, dtype=coffdtype)

			self.Hsigma_onx = np.zeros(self.grid, dtype=coffdtype)
			self.Hsigma_ony = np.zeros(self.grid, dtype=coffdtype)
			self.Hsigma_onz = np.zeros(self.grid, dtype=coffdtype)

			self.Esigma_offx = np.zeros(self.grid, dtype=coffdtype)
			self.Esigma_offy = np.zeros(self.grid, dtype=coffdtype)
			self.Esigma_offz = np.zeros(self.grid, dtype=coffdtype)

			self.Hsigma_offx = np.zeros(self.grid, dtype=coffdtype)
			self.Hsigma_offy = np.zeros(self.grid, dtype=coffdtype)
			self.Hsigma_offz = np.zeros(self.grid, dtype=coffdtype)
			#----------------------------------------------------#

		###############################################################################
		####################### Slices of zgrid that each node got ####################
		###############################################################################
		
		# All nodes should know the slice objects and z-index of all the others
		# as well as its own z-slice object and z-index.
		# Note that rank 0 has zero slice of total grid.
		# Note that rank 0 has no indexes of total grid.
		self.OWN_grid_zslices = [slice(0,0)]
		self.EzHzgrid_zslices = [slice(0,0)]
		self.ExEygrid_zslices = [slice(0,0)]
		self.HxHygrid_zslices = [slice(0,0)]

		self.OWN_grid_zindice = [(0,0)]
		self.EzHzgrid_zindice = [(0,0)]
		self.ExEygrid_zindice = [(0,0)]
		self.HxHygrid_zindice = [(0,0)]

		for rank in range(1,self.size):
			
			if rank == 1 : 
				OWN_grid_zrange = (0,zz  )
				EzHzgrid_zrange = (0,zz  )
				ExEygrid_zrange = (0,zz+1)
				HxHygrid_zrange = (0,zz  )

				OWN_part = slice(OWN_grid_zrange[0], OWN_grid_zrange[1])
				EzHzpart = slice(EzHzgrid_zrange[0], EzHzgrid_zrange[1])
				ExEypart = slice(ExEygrid_zrange[0], ExEygrid_zrange[1])
				HxHypart = slice(HxHygrid_zrange[0], HxHygrid_zrange[1])

				self.OWN_grid_zslices.append(OWN_part) 
				self.EzHzgrid_zslices.append(EzHzpart) 
				self.ExEygrid_zslices.append(ExEypart) 
				self.HxHygrid_zslices.append(HxHypart) 

				self.OWN_grid_zindice.append(OWN_grid_zrange)
				self.EzHzgrid_zindice.append(EzHzgrid_zrange)
				self.ExEygrid_zindice.append(ExEygrid_zrange)
				self.HxHygrid_zindice.append(HxHygrid_zrange)

			elif rank > 1 and rank < (self.size-1) : 

				OWN_grid_zrange = ( (rank-1)*zz  , rank*zz  )
				EzHzgrid_zrange = ( (rank-1)*zz  , rank*zz  )
				ExEygrid_zrange = ( (rank-1)*zz  , rank*zz+1)
				HxHygrid_zrange = ( (rank-1)*zz-1, rank*zz  )

				OWN_part = slice(OWN_grid_zrange[0], OWN_grid_zrange[1])
				EzHzpart = slice(EzHzgrid_zrange[0], EzHzgrid_zrange[1])
				ExEypart = slice(ExEygrid_zrange[0], ExEygrid_zrange[1])
				HxHypart = slice(HxHygrid_zrange[0], HxHygrid_zrange[1])

				self.OWN_grid_zslices.append(OWN_part) 
				self.EzHzgrid_zslices.append(EzHzpart) 
				self.ExEygrid_zslices.append(ExEypart) 
				self.HxHygrid_zslices.append(HxHypart) 

				self.OWN_grid_zindice.append(OWN_grid_zrange)
				self.EzHzgrid_zindice.append(EzHzgrid_zrange)
				self.ExEygrid_zindice.append(ExEygrid_zrange)
				self.HxHygrid_zindice.append(HxHygrid_zrange)

			elif rank == (self.size-1) :

				OWN_grid_zrange = ( (rank-1)*zz  , rank*zz)
				EzHzgrid_zrange = ( (rank-1)*zz  , rank*zz)
				ExEygrid_zrange = ( (rank-1)*zz  , rank*zz)
				HxHygrid_zrange = ( (rank-1)*zz-1, rank*zz)

				OWN_part = slice(OWN_grid_zrange[0], OWN_grid_zrange[1])
				EzHzpart = slice(EzHzgrid_zrange[0], EzHzgrid_zrange[1])
				ExEypart = slice(ExEygrid_zrange[0], ExEygrid_zrange[1])
				HxHypart = slice(HxHygrid_zrange[0], HxHygrid_zrange[1])

				self.OWN_grid_zslices.append(OWN_part) 
				self.EzHzgrid_zslices.append(EzHzpart) 
				self.ExEygrid_zslices.append(ExEypart) 
				self.HxHygrid_zslices.append(HxHypart) 

				self.OWN_grid_zindice.append(OWN_grid_zrange)
				self.EzHzgrid_zindice.append(EzHzgrid_zrange)
				self.ExEygrid_zindice.append(ExEygrid_zrange)
				self.HxHygrid_zindice.append(HxHygrid_zrange)

			else : pass

		if self.rank == 0:

			print("OWN  indice: ", self.OWN_grid_zindice)
			print("EzHz indice: ", self.EzHzgrid_zindice)
			print("ExEy indice: ", self.ExEygrid_zindice)
			print("HxHy indice: ", self.HxHygrid_zindice)

		self.comm.Barrier()
	
	@property
	def gridgap(self): return self._gridgap
	
	@gridgap.setter
	def gridgap(self,gridgap,courant=1./4):

		self._gridgap = gridgap
		self.dx = self._gridgap[0]
		self.dy = self._gridgap[1]
		self.dz = self._gridgap[2]
		self.courant = courant
		self.dt = self.courant * min(self.dx,self.dy,self.dz)/c
		self.maxdt = 2. / c / np.sqrt( (2/self.dz)**2 + (np.pi/self.dx)**2 + (np.pi/self.dy)**2 )

		assert self.dt < self.maxdt , "Time interval is too big so that causality is broken. Lower the courant number."

	def Apply_PML(self,PML):
	
		#########################################################################################################
		############################ Rank 0 calculate coefficient array for PML region ##########################
		#########################################################################################################

		if self.rank == 0:

			self.npml = PML.npml
			self.apply_PML = True

			self.CDx1 = PML.CDx1 ; self.CDx2 = PML.CDx2
			self.CDy1 = PML.CDy1 ; self.CDy2 = PML.CDy2
			self.CDz1 = PML.CDz1 ; self.CDz2 = PML.CDz2

			self.CBx1 = PML.CBx1 ; self.CBx2 = PML.CBx2
			self.CBy1 = PML.CBy1 ; self.CBy2 = PML.CBy2
			self.CBz1 = PML.CBz1 ; self.CBz2 = PML.CBz2

			self.CEx1 = PML.CEx1 ; self.CEx2 = PML.CEx2 ; self.CEx3 = PML.CEx3
			self.CEy1 = PML.CEy1 ; self.CEy2 = PML.CEy2 ; self.CEy3 = PML.CEy3
			self.CEz1 = PML.CEz1 ; self.CEz2 = PML.CEz2 ; self.CEz3 = PML.CEz3

			self.CHx1 = PML.CHx1 ; self.CHx2 = PML.CHx2 ; self.CHx3 = PML.CHx3
			self.CHy1 = PML.CHy1 ; self.CHy2 = PML.CHy2 ; self.CHy3 = PML.CHy3
			self.CHz1 = PML.CHz1 ; self.CHz2 = PML.CHz2 ; self.CHz3 = PML.CHz3

			#########################################################################################
			################## MPI Send Coeff array from rank 0  to rank 1,2,3,... ##################
			#########################################################################################

			for slave in range(1,self.size) :

				EzHz_zslice = self.EzHzgrid_zslices[slave]
				ExEy_zslice = self.ExEygrid_zslices[slave]
				HxHy_zslice = self.HxHygrid_zslices[slave]

				#-----------------------------------#
				#------------- EzHz grid -----------#
				#-----------------------------------#

				send_CDz1 = self.CDz1[:,:,EzHz_zslice]
				send_CDz2 = self.CDz2[:,:,EzHz_zslice]

				send_CBz1 = self.CBz1[:,:,EzHz_zslice]
				send_CBz2 = self.CBz2[:,:,EzHz_zslice]

				send_CEz1 = self.CEz1[:,:,EzHz_zslice]
				send_CEz2 = self.CEz2[:,:,EzHz_zslice]
				send_CEz3 = self.CEz3[:,:,EzHz_zslice]

				send_CHz1 = self.CHz1[:,:,EzHz_zslice]
				send_CHz2 = self.CHz2[:,:,EzHz_zslice]
				send_CHz3 = self.CHz3[:,:,EzHz_zslice]

				#-----------------------------------#
				#------------- ExEy grid -----------#
				#-----------------------------------#

				send_CDx1 = self.CDx1[:,:,ExEy_zslice]
				send_CDx2 = self.CDx2[:,:,ExEy_zslice]
				send_CDy1 = self.CDy1[:,:,ExEy_zslice]
				send_CDy2 = self.CDy2[:,:,ExEy_zslice]

				send_CEx1 = self.CEx1[:,:,ExEy_zslice]
				send_CEx2 = self.CEx2[:,:,ExEy_zslice]
				send_CEx3 = self.CEx3[:,:,ExEy_zslice]
				send_CEy1 = self.CEy1[:,:,ExEy_zslice]
				send_CEy2 = self.CEy2[:,:,ExEy_zslice]
				send_CEy3 = self.CEy3[:,:,ExEy_zslice]

				#-----------------------------------#
				#------------- HxHy grid -----------#
				#-----------------------------------#

				send_CBx1 = self.CBx1[:,:,HxHy_zslice]
				send_CBx2 = self.CBx2[:,:,HxHy_zslice]
				send_CBy1 = self.CBy1[:,:,HxHy_zslice]
				send_CBy2 = self.CBy2[:,:,HxHy_zslice]

				send_CHx1 = self.CHx1[:,:,HxHy_zslice]
				send_CHx2 = self.CHx2[:,:,HxHy_zslice]
				send_CHx3 = self.CHx3[:,:,HxHy_zslice]
				send_CHy1 = self.CHy1[:,:,HxHy_zslice]
				send_CHy2 = self.CHy2[:,:,HxHy_zslice]
				send_CHy3 = self.CHy3[:,:,HxHy_zslice]

				#----------------------------------------------------------#
				#---------- MPI send Coeff array to slave nodes -----------#
				#----------------------------------------------------------#

				self.comm.send( send_CDx1, dest=slave, tag=(slave*100 + 0 ))
				self.comm.send( send_CDx2, dest=slave, tag=(slave*100 + 1 ))
				self.comm.send( send_CBx1, dest=slave, tag=(slave*100 + 2 ))
				self.comm.send( send_CBx2, dest=slave, tag=(slave*100 + 3 ))

				self.comm.send( send_CDy1, dest=slave, tag=(slave*100 + 4 ))
				self.comm.send( send_CDy2, dest=slave, tag=(slave*100 + 5 ))
				self.comm.send( send_CBy1, dest=slave, tag=(slave*100 + 6 ))
				self.comm.send( send_CBy2, dest=slave, tag=(slave*100 + 7 ))

				self.comm.send( send_CDz1, dest=slave, tag=(slave*100 + 8 ))
				self.comm.send( send_CDz2, dest=slave, tag=(slave*100 + 9 ))
				self.comm.send( send_CBz1, dest=slave, tag=(slave*100 + 10))
				self.comm.send( send_CBz2, dest=slave, tag=(slave*100 + 11))

				self.comm.send( send_CEx1, dest=slave, tag=(slave*100 + 12))
				self.comm.send( send_CEx2, dest=slave, tag=(slave*100 + 13))
				self.comm.send( send_CEx3, dest=slave, tag=(slave*100 + 14))

				self.comm.send( send_CHx1, dest=slave, tag=(slave*100 + 15))
				self.comm.send( send_CHx2, dest=slave, tag=(slave*100 + 16))
				self.comm.send( send_CHx3, dest=slave, tag=(slave*100 + 17))

				self.comm.send( send_CEy1, dest=slave, tag=(slave*100 + 18))
				self.comm.send( send_CEy2, dest=slave, tag=(slave*100 + 19))
				self.comm.send( send_CEy3, dest=slave, tag=(slave*100 + 20))

				self.comm.send( send_CHy1, dest=slave, tag=(slave*100 + 21))
				self.comm.send( send_CHy2, dest=slave, tag=(slave*100 + 22))
				self.comm.send( send_CHy3, dest=slave, tag=(slave*100 + 23))

				self.comm.send( send_CEz1, dest=slave, tag=(slave*100 + 24))
				self.comm.send( send_CEz2, dest=slave, tag=(slave*100 + 25))
				self.comm.send( send_CEz3, dest=slave, tag=(slave*100 + 26))

				self.comm.send( send_CHz1, dest=slave, tag=(slave*100 + 27))
				self.comm.send( send_CHz2, dest=slave, tag=(slave*100 + 28))
				self.comm.send( send_CHz3, dest=slave, tag=(slave*100 + 29))

		else : # rank 1,2,3,... receives coefficient array from rank 0

			#-------------------------------------------------------------#
			#---------- MPI recv Coeff array from master nodes -----------#
			#-------------------------------------------------------------#

			self.CDx1 = self.comm.recv( source=0, tag=(self.rank*100 + 0 ))
			self.CDx2 = self.comm.recv( source=0, tag=(self.rank*100 + 1 ))
			self.CBx1 = self.comm.recv( source=0, tag=(self.rank*100 + 2 ))
			self.CBx2 = self.comm.recv( source=0, tag=(self.rank*100 + 3 ))

			self.CDy1 = self.comm.recv( source=0, tag=(self.rank*100 + 4 ))
			self.CDy2 = self.comm.recv( source=0, tag=(self.rank*100 + 5 ))
			self.CBy1 = self.comm.recv( source=0, tag=(self.rank*100 + 6 ))
			self.CBy2 = self.comm.recv( source=0, tag=(self.rank*100 + 7 ))

			self.CDz1 = self.comm.recv( source=0, tag=(self.rank*100 + 8 ))
			self.CDz2 = self.comm.recv( source=0, tag=(self.rank*100 + 9 ))
			self.CBz1 = self.comm.recv( source=0, tag=(self.rank*100 + 10))
			self.CBz2 = self.comm.recv( source=0, tag=(self.rank*100 + 11))

			self.CEx1 = self.comm.recv( source=0, tag=(self.rank*100 + 12))
			self.CEx2 = self.comm.recv( source=0, tag=(self.rank*100 + 13))
			self.CEx3 = self.comm.recv( source=0, tag=(self.rank*100 + 14))

			self.CHx1 = self.comm.recv( source=0, tag=(self.rank*100 + 15))
			self.CHx2 = self.comm.recv( source=0, tag=(self.rank*100 + 16))
			self.CHx3 = self.comm.recv( source=0, tag=(self.rank*100 + 17))

			self.CEy1 = self.comm.recv( source=0, tag=(self.rank*100 + 18))
			self.CEy2 = self.comm.recv( source=0, tag=(self.rank*100 + 19))
			self.CEy3 = self.comm.recv( source=0, tag=(self.rank*100 + 20))

			self.CHy1 = self.comm.recv( source=0, tag=(self.rank*100 + 21))
			self.CHy2 = self.comm.recv( source=0, tag=(self.rank*100 + 22))
			self.CHy3 = self.comm.recv( source=0, tag=(self.rank*100 + 23))

			self.CEz1 = self.comm.recv( source=0, tag=(self.rank*100 + 24))
			self.CEz2 = self.comm.recv( source=0, tag=(self.rank*100 + 25))
			self.CEz3 = self.comm.recv( source=0, tag=(self.rank*100 + 26))

			self.CHz1 = self.comm.recv( source=0, tag=(self.rank*100 + 27))
			self.CHz2 = self.comm.recv( source=0, tag=(self.rank*100 + 28))
			self.CHz3 = self.comm.recv( source=0, tag=(self.rank*100 + 29))

			print("rank {0} recieved coefficient array from rank 0 successfully." .format(self.rank))

class Fields(object):

	def __init__(self,Space):
		"""Set up Fields in the space.

		Parameters
		-----------
		Space : Space object
			field object is set upon the info of the space user had set
			
		"""

		self.comm   = Space.comm
		self.rank   = Space.rank
		self.size   = Space.size
		self.dtype  = Space.dtype
		self.grid   = Space.grid

		self.courant  = Space.courant
		self.hostname = Space.hostname

		self.OWN_grid_zslices  = Space.OWN_grid_zslices
		self.EzHzgrid_zslices  = Space.EzHzgrid_zslices
		self.ExEygrid_zslices  = Space.ExEygrid_zslices
		self.HxHygrid_zslices  = Space.HxHygrid_zslices

		self.OWN_grid_zindice = Space.OWN_grid_zindice
		self.EzHzgrid_zindice = Space.EzHzgrid_zindice
		self.ExEygrid_zindice = Space.ExEygrid_zindice
		self.HxHygrid_zindice = Space.HxHygrid_zindice

		self.dt    = Space.dt
		self.gridx = Space.gridx ; self.dx = Space.dx ; self.gridxc = Space.gridxc
		self.gridy = Space.gridy ; self.dy = Space.dy ; self.gridyc = Space.gridyc
		self.gridz = Space.gridz ; self.dz = Space.dz ; self.gridzc = Space.gridzc
		
		if self.rank > 0 :

			self.OWN_grid_per_node = Space.OWN_grid_per_node
			self.EzHzgrid_per_node = Space.EzHzgrid_per_node
			self.ExEygrid_per_node = Space.ExEygrid_per_node
			self.HxHygrid_per_node = Space.HxHygrid_per_node

			# Fields with EzHz grid
			self.Dz = np.zeros(Space.EzHzgrid_per_node, dtype=Space.dtype)
			self.Ez = np.zeros(Space.EzHzgrid_per_node, dtype=Space.dtype)
			self.Bz = np.zeros(Space.EzHzgrid_per_node, dtype=Space.dtype)
			self.Hz = np.zeros(Space.EzHzgrid_per_node, dtype=Space.dtype)

			# Fields with ExEy grid
			self.Dx = np.zeros(Space.ExEygrid_per_node, dtype=Space.dtype)
			self.Ex = np.zeros(Space.ExEygrid_per_node, dtype=Space.dtype)
			self.Dy = np.zeros(Space.ExEygrid_per_node, dtype=Space.dtype)
			self.Ey = np.zeros(Space.ExEygrid_per_node, dtype=Space.dtype)

			# Fields with HxHy grid
			self.Bx = np.zeros(Space.HxHygrid_per_node, dtype=Space.dtype)
			self.Hx = np.zeros(Space.HxHygrid_per_node, dtype=Space.dtype)
			self.By = np.zeros(Space.HxHygrid_per_node, dtype=Space.dtype)
			self.Hy = np.zeros(Space.HxHygrid_per_node, dtype=Space.dtype)

			# Take Coefficient array as attributes.
			self.CDx1 = Space.CDx1 ; self.CDx2 = Space.CDx2
			self.CDy1 = Space.CDy1 ; self.CDy2 = Space.CDy2
			self.CDz1 = Space.CDz1 ; self.CDz2 = Space.CDz2

			self.CBx1 = Space.CBx1 ; self.CBx2 = Space.CBx2
			self.CBy1 = Space.CBy1 ; self.CBy2 = Space.CBy2
			self.CBz1 = Space.CBz1 ; self.CBz2 = Space.CBz2

			self.CEx1 = Space.CEx1 ; self.CEx2 = Space.CEx2 ; self.CEx3 = Space.CEx3
			self.CEy1 = Space.CEy1 ; self.CEy2 = Space.CEy2 ; self.CEy3 = Space.CEy3
			self.CEz1 = Space.CEz1 ; self.CEz2 = Space.CEz2 ; self.CEz3 = Space.CEz3

			self.CHx1 = Space.CHx1 ; self.CHx2 = Space.CHx2 ; self.CHx3 = Space.CHx3
			self.CHy1 = Space.CHy1 ; self.CHy2 = Space.CHy2 ; self.CHy3 = Space.CHy3
			self.CHz1 = Space.CHz1 ; self.CHz2 = Space.CHz2 ; self.CHz3 = Space.CHz3

		elif self.rank == 0 :
			
			####### rank 0 has no field array #######

			self.Dx = np.zeros(1) ; self.Ex = np.zeros(1)
			self.Dy = np.zeros(1) ; self.Ey = np.zeros(1)
			self.Dz = np.zeros(1) ; self.Ez = np.zeros(1)

			self.Bx = np.zeros(1) ; self.Hx = np.zeros(1)
			self.By = np.zeros(1) ; self.Hy = np.zeros(1)
			self.Bz = np.zeros(1) ; self.Hz = np.zeros(1)
			
	@property
	def nsteps(self): return self._nsteps

	@nsteps.setter
	def nsteps(self, nsteps): 
		self._nsteps = nsteps
		return

	@property
	def ref_trs_pos(self): return self.ref_pos, self.trs_pos

	@ref_trs_pos.setter
	def ref_trs_pos(self, pos):
		"""Set z position to collect ref and trs

		PARAMETERS
		----------
		pos : tuple or list
				z index of ref position and trs position

		RETURNS
		-------
		None
		"""

		#assert type(pos[0]) == type(int(0))
		#assert type(pos[1]) == type(int(0))

		if pos[0] >= 0: self.ref_pos = pos[0]
		else          : self.ref_pos = pos[0] + self.grid[2]
		if pos[1] >= 0: self.trs_pos = pos[1]
		else          : self.trs_pos = pos[1] + self.grid[2]

		##################################################################################
		######################## All rank should know who gets trs #######################
		##################################################################################

		for rank in range(self.size) : 

			start = self.OWN_grid_zindice[rank][0]
			end   = self.OWN_grid_zindice[rank][1]

			if self.trs_pos >= start and self.trs_pos < end : 
				self.who_get_trs     = rank 
				self.trs_pos_in_node = self.trs_pos - start

		###################################################################################
		####################### All rank should know who gets the ref #####################
		###################################################################################

		for rank in range(self.size):
			start = self.OWN_grid_zindice[rank][0]
			end   = self.OWN_grid_zindice[rank][1]

			if self.ref_pos >= start and self.ref_pos < end :
				self.who_get_ref     = rank
				self.ref_pos_in_node = self.ref_pos - start 

		#---------------------------------------------------------------------------------#
		#----------------------- Ready to put ref and trs collector ----------------------#
		#---------------------------------------------------------------------------------#

		if   self.rank == self.who_get_trs:
			print("rank %d: I collect trs from %d which is essentially %d in my own grid."\
					 %(self.rank, self.trs_pos, self.trs_pos_in_node))
			self.trs = np.zeros(self._nsteps, dtype=self.dtype) 

		if self.rank == self.who_get_ref: 
			print("rank %d: I collect ref from %d which is essentially %d in my own grid."\
					 %(self.rank, self.ref_pos, self.ref_pos_in_node))
			self.ref = np.zeros(self._nsteps, dtype=self.dtype)

		if self.rank == 0:
			# This arrays are necessary for rank0 to collect src,trs and ref from slave node.
			self.src = np.zeros(self._nsteps, dtype=self.dtype)
			self.trs = np.zeros(self._nsteps, dtype=self.dtype)
			self.ref = np.zeros(self._nsteps, dtype=self.dtype)

		else : pass

		self.comm.Barrier()

	def set_src(self, where, position, put_type):
		"""Set the position, type of the source and field.

		PARAMETERS
		----------
		where : string
			The field to pur source

		position : list
			A list which has int or slice object as its element.
			The elements defines the position of the source in the field.

		src_type : string or tuple
			Specify the source type. There are three types of the source.
			Plane wave, point source and line source.
			If you want to put plane wave or line source, the argument type should be a tuple
			which the first element is 'plane' or 'line'
			and second element is the direction to say 'x','y' or 'z'.

		put_type : string
			'soft' or 'hard'

		RETURNS
		-------
		None
		"""

		self.put_type = put_type
		self.where = where

		#------------------------------------------------------------#
		#---------- All rank should know who puts the src -----------#
		#------------------------------------------------------------#

		assert len(position) == 3, "position argument is a list or tuple with length 3."

		self.src_xpos = position[0]
		self.src_ypos = position[1]
		self.src_zpos = position[2]

		if type(self.src_zpos) == int :

			for rank in range(1,self.size):
				start = self.OWN_grid_zindice[rank][0]
				end   = self.OWN_grid_zindice[rank][1]

				if self.src_zpos >= start and self.src_zpos < end:
					self.who_put_src	  = rank
					self.src_zpos_in_node = self.src_zpos - start

					if self.rank == self.who_put_src:
						self.src = np.zeros(self._nsteps, dtype=self.dtype)
						print("rank %d: I put src at %d which is essentially %d in my own grid."\
								 %(self.rank, self.src_zpos, self.src_zpos_in_node))

		elif self.src_zpos == slice(None,None):
			self.who_put_src = self.rank
			
			if   self.who_put_src >  1: self.src_zpos_in_node = slice(1,None)
			elif self.who_put_src == 1: self.src_zpos_in_node = slice(None,None)

		self.comm.Barrier()

		return None

	def put_src(self, pulse):

		###############################################################################
		###################### Put the source into designated field ###################
		###############################################################################

		self.pulse_value = self.dtype(pulse)

		if self.rank == self.who_put_src and self.rank > 0:

			x = self.src_xpos
			y = self.src_ypos
			z = self.src_zpos_in_node

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
		
		ft  = np.fft.fftn
		ift = np.fft.ifftn
		nax = np.newaxis

		#--------------------------------------------------------------#
		#------------ MPI send Ex and Ey to previous rank -------------#
		#--------------------------------------------------------------#

		if self.rank > 1 and self.rank < self.size: # rank 2,3,...,n-1

			send_Ex_first = self.Ex[:,:,0].copy()
			send_Ey_first = self.Ey[:,:,0].copy()

			self.comm.send( send_Ex_first, dest=(self.rank-1), tag=(tstep*100+4))
			self.comm.send( send_Ey_first, dest=(self.rank-1), tag=(tstep*100+5))

		else : pass

		#--------------------------------------------------------------#
		#------------ MPI recv Ex and Ey from next rank ---------------#
		#--------------------------------------------------------------#

		if self.rank > 0 and self.rank < (self.size-1): # rank 1,2,...,n-2

			recv_Ex_last = self.comm.recv( source=(self.rank+1), tag=(tstep*100+4))
			recv_Ey_last = self.comm.recv( source=(self.rank+1), tag=(tstep*100+5))

			self.Ex[:,:,-1] = recv_Ex_last
			self.Ey[:,:,-1] = recv_Ey_last

		else : pass

	#	print(self.rank, tstep, "Here?")

		# First slave node.
		if self.rank == 1:

			xx = self.HxHygrid_per_node[0]
			yy = self.HxHygrid_per_node[1]
			zz = self.HxHygrid_per_node[2]

			# Update Hx
			for k in range(zz):
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
			for k in range(zz):
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
			zz = self.EzHzgrid_per_node[2]

			for k in range(zz):
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
		if self.rank > 1 and self.rank < (self.size-1):

			xx = self.HxHygrid_per_node[0]
			yy = self.HxHygrid_per_node[1]
			zz = self.HxHygrid_per_node[2]

			# Update Hx
			for k in range(1,zz):
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
			for k in range(1,zz):
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
			zz = self.EzHzgrid_per_node[2]

			for k in range(zz):
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
		if self.rank == (self.size-1):

			xx = self.HxHygrid_per_node[0]
			yy = self.HxHygrid_per_node[1]
			zz = self.HxHygrid_per_node[2]

			# Update Hx
			for k in range(1,zz-1):
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
			for k in range(1,zz-1):
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
			zz = self.EzHzgrid_per_node[2]

			for k in range(zz):
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

		if self.rank > 0 and self.rank < (self.size-1): # rank 1,2,3,...,n-2

			send_Hx_last = self.Hx[:,:,-1].copy()
			send_Hy_last = self.Hy[:,:,-1].copy()

			self.comm.send(send_Hx_last, dest=(self.rank+1), tag=(tstep*100+1))
			self.comm.send(send_Hy_last, dest=(self.rank+1), tag=(tstep*100+2))

		else : pass

		#---------------------------------------------------------#
		#--------- MPI recv Hx and Hy from previous rank ---------#
		#---------------------------------------------------------#

		if self.rank > 1 and self.rank < self.size: # rank 2,3,...,n-1

			recv_Hx_first = self.comm.recv( source=(self.rank-1), tag=(tstep*100+1))
			recv_Hy_first = self.comm.recv( source=(self.rank-1), tag=(tstep*100+2) )

			self.Hx[:,:,0] = recv_Hx_first
			self.Hy[:,:,0] = recv_Hy_first
		
		else : pass

		# First slave node.
		if self.rank == 1:

			xx = self.ExEygrid_per_node[0]
			yy = self.ExEygrid_per_node[1]
			zz = self.ExEygrid_per_node[2]

			# Update Ex
			for k in range(1,zz-1):
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
			for k in range(1,zz-1):
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
			zz = self.EzHzgrid_per_node[2]

			# Update Ez
			for k in range(zz):
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
		if self.rank > 1 and self.rank < (self.size-1):

			xx = self.ExEygrid_per_node[0]
			yy = self.ExEygrid_per_node[1]
			zz = self.ExEygrid_per_node[2]

			# Update Ex
			for k in range(zz-1):
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
			for k in range(zz-1):
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
			zz = self.EzHzgrid_per_node[2]

			# Update Ez
			for k in range(zz):
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
		if self.rank == (self.size-1):

			xx = self.ExEygrid_per_node[0]
			yy = self.ExEygrid_per_node[1]
			zz = self.ExEygrid_per_node[2]

			# Update Ex
			for k in range(zz-1):
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
			for k in range(zz-1):
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
			zz = self.EzHzgrid_per_node[2]

			# Update Ez
			for k in range(zz):
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

		if self.rank == self.who_get_ref :
			
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
			
		if self.rank == self.who_get_trs :
			
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
