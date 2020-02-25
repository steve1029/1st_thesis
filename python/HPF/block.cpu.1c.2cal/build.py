import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time, os, datetime, sys, platform
from mpi4py import MPI
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, mu_0, epsilon_0

class Space(object):
	
	def __init__(self,**kwargs):

		self.dimension = 3
		self.dtype = np.complex128
		
		self.comm = MPI.COMM_WORLD
		self.rank = self.comm.Get_rank()
		self.size = self.comm.Get_size()
		self.hostname = MPI.Get_processor_name()

		assert sys.version_info >= (3,4,0)
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
			self.OWNgrid_per_node = [self.gridx, self.gridy, zz  ]
			self.EEHgrid_per_node = [self.gridx, self.gridy, zz+1]
			self.HHEgrid_per_node = [self.gridx, self.gridy, zz  ]

		elif self.rank > 1 and self.rank < (self.size-1): 
			self.OWNgrid_per_node = [self.gridx, self.gridy, zz  ]
			self.EEHgrid_per_node = [self.gridx, self.gridy, zz+1]
			self.HHEgrid_per_node = [self.gridx, self.gridy, zz+1]

		elif self.rank == (self.size-1):
			self.OWNgrid_per_node = [self.gridx, self.gridy, zz  ]
			self.EEHgrid_per_node = [self.gridx, self.gridy, zz  ]
			self.HHEgrid_per_node = [self.gridx, self.gridy, zz+1]
	
		if self.rank > 0:
			 print("rank: %d, name: %s, has EEH grid %s and HHE grid %s \
				" %(self.rank, self.hostname,tuple(self.EEHgrid_per_node),tuple(self.HHEgrid_per_node)))

		if self.rank == 0 :

			if   self.dtype == np.complex128: coffdtype = np.float64
			elif self.dtype == np.complex64 : coffdtype = np.float32

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
		
		# All nodes should have the slice object and indexes of each slave node.
		# rank 0 has zero slice of total grid.
		# rank 0 has no indexes of total grid.

		self.OWNslices_of_slaves  = [slice(0,0)]
		self.EEHslices_of_slaves  = [slice(0,0)]
		self.HHEslices_of_slaves  = [slice(0,0)]

		self.OWNindexes_of_slaves = [(0,0)]
		self.EEHindexes_of_slaves = [(0,0)]
		self.HHEindexes_of_slaves = [(0,0)]

		for slave in range(1,self.size):
			
			if slave == 1 : 
				OWNindex = (0,zz  )
				EEHindex = (0,zz+1)
				HHEindex = (0,zz  )

				OWNpart = slice(OWNindex[0],OWNindex[1])
				EEHpart = slice(EEHindex[0],EEHindex[1])
				HHEpart = slice(HHEindex[0],HHEindex[1])

				self.OWNslices_of_slaves.append(OWNpart) 
				self.EEHslices_of_slaves.append(EEHpart) 
				self.HHEslices_of_slaves.append(HHEpart) 

				self.OWNindexes_of_slaves.append(OWNindex)
				self.EEHindexes_of_slaves.append(EEHindex)
				self.HHEindexes_of_slaves.append(HHEindex)

			elif slave > 1 and slave < (self.size-1) : 

				OWNindex = ( (slave-1)*zz  , slave*zz  )
				EEHindex = ( (slave-1)*zz  , slave*zz+1)
				HHEindex = ( (slave-1)*zz-1, slave*zz  )

				OWNpart = slice(OWNindex[0],OWNindex[1])
				EEHpart = slice(EEHindex[0],EEHindex[1])
				HHEpart = slice(HHEindex[0],HHEindex[1])

				self.OWNslices_of_slaves.append(OWNpart) 
				self.EEHslices_of_slaves.append(EEHpart) 
				self.HHEslices_of_slaves.append(HHEpart) 

				self.OWNindexes_of_slaves.append(OWNindex)
				self.EEHindexes_of_slaves.append(EEHindex)
				self.HHEindexes_of_slaves.append(HHEindex)

			elif slave == (self.size-1) :

				OWNindex = ( (slave-1)*zz  , slave*zz)
				EEHindex = ( (slave-1)*zz  , slave*zz)
				HHEindex = ( (slave-1)*zz-1, slave*zz)

				OWNpart = slice(OWNindex[0],OWNindex[1])
				EEHpart = slice(EEHindex[0],EEHindex[1])
				HHEpart = slice(HHEindex[0],HHEindex[1])

				self.OWNslices_of_slaves.append(OWNpart) 
				self.EEHslices_of_slaves.append(EEHpart) 
				self.HHEslices_of_slaves.append(HHEpart) 

				self.OWNindexes_of_slaves.append(OWNindex)
				self.EEHindexes_of_slaves.append(EEHindex)
				self.HHEindexes_of_slaves.append(HHEindex)

			else : pass

		assert len(self.OWNslices_of_slaves)  == self.size
		assert len(self.EEHslices_of_slaves)  == self.size
		assert len(self.HHEslices_of_slaves)  == self.size
		assert len(self.OWNindexes_of_slaves) == self.size
		assert len(self.EEHindexes_of_slaves) == self.size
		assert len(self.HHEindexes_of_slaves) == self.size

		if self.rank == 0:

			print("OWN indexes: ", self.OWNindexes_of_slaves)
			print("EEH indexes: ", self.EEHindexes_of_slaves)
			print("HHE indexes: ", self.HHEindexes_of_slaves)

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

				EEHzslice = self.EEHslices_of_slaves[slave]
				HHEzslice = self.HHEslices_of_slaves[slave]

				#----------------------------------#
				#------------ EEH grid ------------#
				#----------------------------------#

				send_CDx1 = self.CDx1[:,:,EEHzslice]
				send_CDx2 = self.CDx2[:,:,EEHzslice]
				send_CDy1 = self.CDy1[:,:,EEHzslice]
				send_CDy2 = self.CDy2[:,:,EEHzslice]
				send_CBz1 = self.CBz1[:,:,EEHzslice]
				send_CBz2 = self.CBz2[:,:,EEHzslice]

				send_CEx1 = self.CEx1[:,:,EEHzslice]
				send_CEx2 = self.CEx2[:,:,EEHzslice]
				send_CEx3 = self.CEx3[:,:,EEHzslice]
				send_CEy1 = self.CEy1[:,:,EEHzslice]
				send_CEy2 = self.CEy2[:,:,EEHzslice]
				send_CEy3 = self.CEy3[:,:,EEHzslice]
				send_CHz1 = self.CHz1[:,:,EEHzslice]
				send_CHz2 = self.CHz2[:,:,EEHzslice]
				send_CHz3 = self.CHz3[:,:,EEHzslice]

				#----------------------------------#
				#------------ HHE grid ------------#
				#----------------------------------#

				send_CBx1 = self.CBx1[:,:,HHEzslice]
				send_CBx2 = self.CBx2[:,:,HHEzslice]
				send_CBy1 = self.CBy1[:,:,HHEzslice]
				send_CBy2 = self.CBy2[:,:,HHEzslice]
				send_CDz1 = self.CDz1[:,:,HHEzslice]
				send_CDz2 = self.CDz2[:,:,HHEzslice]

				send_CHx1 = self.CHx1[:,:,HHEzslice]
				send_CHx2 = self.CHx2[:,:,HHEzslice]
				send_CHx3 = self.CHx3[:,:,HHEzslice]
				send_CHy1 = self.CHy1[:,:,HHEzslice]
				send_CHy2 = self.CHy2[:,:,HHEzslice]
				send_CHy3 = self.CHy3[:,:,HHEzslice]
				send_CEz1 = self.CEz1[:,:,HHEzslice]
				send_CEz2 = self.CEz2[:,:,HHEzslice]
				send_CEz3 = self.CEz3[:,:,HHEzslice]

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

			#---------------------------------------------------------------------------------------#
			#------------------------------------ EEH grid check -----------------------------------#
			#---------------------------------------------------------------------------------------#

			assert np.all(self.CDx1) == True ; assert self.CDx1.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CDx2) == True ; assert self.CDx2.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CDy1) == True ; assert self.CDy1.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CDy2) == True ; assert self.CDy2.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CBz1) == True ; assert self.CBz1.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CBz2) == True ; assert self.CBz2.shape == tuple(self.EEHgrid_per_node)

			assert np.all(self.CEx1) == True ; assert self.CEx1.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CEx2) == True ; assert self.CEx2.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CEx3) == True ; assert self.CEx3.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CEy1) == True ; assert self.CEy1.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CEy2) == True ; assert self.CEy2.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CEy3) == True ; assert self.CEy3.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CHz1) == True ; assert self.CHz1.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CHz2) == True ; assert self.CHz2.shape == tuple(self.EEHgrid_per_node)
			assert np.all(self.CHz3) == True ; assert self.CHz3.shape == tuple(self.EEHgrid_per_node)

			#---------------------------------------------------------------------------------------#
			#------------------------------------ HHE grid check -----------------------------------#
			#---------------------------------------------------------------------------------------#

			assert np.all(self.CBx1) == True ; assert self.CBx1.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CBx2) == True ; assert self.CBx2.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CBy1) == True ; assert self.CBy1.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CBy2) == True ; assert self.CBy2.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CDz1) == True ; assert self.CDz1.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CDz2) == True ; assert self.CDz2.shape == tuple(self.HHEgrid_per_node)

			assert np.all(self.CHx1) == True ; assert self.CHx1.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CHx2) == True ; assert self.CHx2.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CHx3) == True ; assert self.CHx3.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CHy1) == True ; assert self.CHy1.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CHy2) == True ; assert self.CHy2.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CHy3) == True ; assert self.CHy3.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CEz1) == True ; assert self.CEz1.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CEz2) == True ; assert self.CEz2.shape == tuple(self.HHEgrid_per_node)
			assert np.all(self.CEz3) == True ; assert self.CEz3.shape == tuple(self.HHEgrid_per_node)

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

		self.OWNslices_of_slaves  = Space.OWNslices_of_slaves
		self.EEHslices_of_slaves  = Space.EEHslices_of_slaves
		self.HHEslices_of_slaves  = Space.HHEslices_of_slaves

		self.OWNindexes_of_slaves = Space.OWNindexes_of_slaves
		self.EEHindexes_of_slaves = Space.EEHindexes_of_slaves
		self.HHEindexes_of_slaves = Space.HHEindexes_of_slaves

		self.dt    = Space.dt
		self.gridx = Space.gridx ; self.dx = Space.dx ; self.gridxc = Space.gridxc
		self.gridy = Space.gridy ; self.dy = Space.dy ; self.gridyc = Space.gridyc
		self.gridz = Space.gridz ; self.dz = Space.dz ; self.gridzc = Space.gridzc
		
		if self.rank > 0 :

			self.OWNgrid_per_node = Space.OWNgrid_per_node
			self.EEHgrid_per_node = Space.EEHgrid_per_node
			self.HHEgrid_per_node = Space.HHEgrid_per_node

			xyplane = (self.OWNgrid_per_node[0], self.OWNgrid_per_node[1])

			EEHones  = np.ones (Space.EEHgrid_per_node, dtype=Space.dtype)
			HHEones  = np.ones (Space.HHEgrid_per_node, dtype=Space.dtype)

			EEHzeros = np.zeros(Space.EEHgrid_per_node, dtype=Space.dtype)
			HHEzeros = np.zeros(Space.HHEgrid_per_node, dtype=Space.dtype)

			self.kx = np.fft.fftfreq(Space.gridx, Space.dx) * 2. * np.pi
			self.ky = np.fft.fftfreq(Space.gridy, Space.dy) * 2. * np.pi

			nax = np.newaxis
			self.ikx = 1.j * self.kx[:,nax] * np.ones(xyplane, dtype=Space.dtype)
			self.iky = 1.j * self.ky[nax,:] * np.ones(xyplane, dtype=Space.dtype) 

			self.EEHikx = 1.j * self.kx[:,nax,nax] * EEHones.copy() 
			self.EEHiky = 1.j * self.ky[nax,:,nax] * EEHones.copy()

			self.HHEikx = 1.j * self.kx[:,nax,nax] * HHEones.copy() 
			self.HHEiky = 1.j * self.ky[nax,:,nax] * HHEones.copy()

			# Fields with EEH grid
			self.Dx = EEHzeros.copy(); self.Ex = EEHzeros.copy()
			self.Dy = EEHzeros.copy(); self.Ey = EEHzeros.copy()
			self.Bz = EEHzeros.copy(); self.Hz = EEHzeros.copy()

			# Fields with HHE grid
			self.Bx = HHEzeros.copy(); self.Hx = HHEzeros.copy()
			self.By = HHEzeros.copy(); self.Hy = HHEzeros.copy()
			self.Dz = HHEzeros.copy(); self.Ez = HHEzeros.copy()

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

			start = self.OWNindexes_of_slaves[rank][0]
			end   = self.OWNindexes_of_slaves[rank][1]

			if self.trs_pos >= start and self.trs_pos < end : 
				self.who_get_trs     = rank 
				self.trs_pos_in_node = self.trs_pos - start

		###################################################################################
		####################### All rank should know who gets the ref #####################
		###################################################################################

		for rank in range(self.size):
			start = self.OWNindexes_of_slaves[rank][0]
			end   = self.OWNindexes_of_slaves[rank][1]

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
				start = self.OWNindexes_of_slaves[rank][0]
				end   = self.OWNindexes_of_slaves[rank][1]

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
		"""By defining internal function Hxupdater, Hy updater and Hz updater, this method
		update the total H field which is distributed in slave nodes.

		PARAMETERS
		----------
		tstep	:	int
			time step of the simulation.

		RETURNS
		-------
		None

		"""

		def Hzupdater(start, end, QBz,QHz):

			if self.rank == 1:

				for k in range(start,end):

					previous_k = self.Bz[:,:,k].copy()

					CBz1_k = self.CBz1[:,:,k]
					CBz2_k = self.CBz2[:,:,k]
					CHz1_k = self.CHz1[:,:,k]
					CHz2_k = self.CHz2[:,:,k]
					CHz3_k = self.CHz3[:,:,k]

					diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
					diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
					self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k - diffyEx_k)
					self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

#				previous = self.Bz.copy()
#
#				CBz1 = self.CBz1
#				CBz2 = self.CBz2
#				CHz1 = self.CHz1
#				CHz2 = self.CHz2
#				CHz3 = self.CHz3
#
#				diffxEy = ift(self.EEHikx * ft(self.Ey, axes=(0,)), axes=(0,))
#				diffyEx = ift(self.EEHiky * ft(self.Ex, axes=(1,)), axes=(1,))
#				self.Bz = CBz1 * self.Bz + CBz2 * (diffxEy - diffyEx)
#				self.Hz = CHz1 * self.Hz + CHz2 * self.Bz + CHz3 * previous
#
				QBz.put(self.Bz[:,:,start:end])
				QHz.put(self.Hz[:,:,start:end])

			elif self.rank > 1 and self.rank < (self.size-1):

				for k in range(start, end):

					previous_k = self.Bz[:,:,k].copy()

					CBz1_k = self.CBz1[:,:,k]
					CBz2_k = self.CBz2[:,:,k]
					CHz1_k = self.CHz1[:,:,k]
					CHz2_k = self.CHz2[:,:,k]
					CHz3_k = self.CHz3[:,:,k]

					diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
					diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
					self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k - diffyEx_k)
					self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

#				previous = self.Bz.copy()
#
#				CBz1 = self.CBz1
#				CBz2 = self.CBz2
#				CHz1 = self.CHz1
#				CHz2 = self.CHz2
#				CHz3 = self.CHz3
#
#				diffxEy = ift(self.EEHikx * ft(self.Ey, axes=(0,)), axes=(0,))
#				diffyEx = ift(self.EEHiky * ft(self.Ex, axes=(1,)), axes=(1,))
#				self.Bz = CBz1 * self.Bz + CBz2 * (diffxEy - diffyEx)
#				self.Hz = CHz1 * self.Hz + CHz2 * self.Bz + CHz3 * previous
#
				QBz.put(self.Bz[:,:,start:end])
				QHz.put(self.Hz[:,:,start:end])

			elif self.rank == (self.size-1):

				for k in range(start,end):

					previous_k = self.Bz[:,:,k].copy()

					CBz1_k = self.CBz1[:,:,k]
					CBz2_k = self.CBz2[:,:,k]
					CHz1_k = self.CHz1[:,:,k]
					CHz2_k = self.CHz2[:,:,k]
					CHz3_k = self.CHz3[:,:,k]

					diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
					diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
					self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k- diffyEx_k)
					self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

#				previous = self.Bz.copy()
#
#				CBz1 = self.CBz1
#				CBz2 = self.CBz2
#				CHz1 = self.CHz1
#				CHz2 = self.CHz2
#				CHz3 = self.CHz3
#
#				diffxEy = ift(self.EEHikx * ft(self.Ey, axes=(0,)), axes=(0,))
#				diffyEx = ift(self.EEHiky * ft(self.Ex, axes=(1,)), axes=(1,))
#				self.Bz = CBz1 * self.Bz + CBz2 * (diffxEy - diffyEx)
#				self.Hz = CHz1 * self.Hz + CHz2 * self.Bz + CHz3 * previous
#
				QBz.put(self.Bz[:,:,start:end])
				QHz.put(self.Hz[:,:,start:end])
#			print("rank: {}, tstep : {}, Hz updater : {}, time: {}" .format(self.rank, tstep, mp.current_process(), datetime.datetime.now()))

			return

		def Hxupdater(start, end, QBx, QHx):

			# First slave node
			if self.rank == 1:

				for k in range(start,end):

					previous = self.Bx[:,:,k].copy()

					CBx1 = self.CBx1[:,:,k]
					CBx2 = self.CBx2[:,:,k]
					CHx1 = self.CHx1[:,:,k]
					CHx2 = self.CHx2[:,:,k]
					CHx3 = self.CHx3[:,:,k]

					diffzEy_k = (self.Ey[:,:,k+1] - self.Ey[:,:,k]) / self.dz 
					diffyEz_k = ift(self.iky * ft(self.Ez[:,:,k], axes=(1,)), axes=(1,))

					self.Bx[:,:,k] = CBx1 * self.Bx[:,:,k] + CBx2 * (diffyEz_k - diffzEy_k)
					self.Hx[:,:,k] = CHx1 * self.Hx[:,:,k] + CHx2 * self.Bx[:,:,k] + CHx3 * previous

				QBx.put(self.Bx[:,:,start:end])
				QHx.put(self.Hx[:,:,start:end])

			# Middle slave node
			elif self.rank > 1 and self.rank < (self.size-1):

				for k in range(start,end):

					previous = self.Bx[:,:,k].copy()

					CBx1 = self.CBx1[:,:,k]
					CBx2 = self.CBx2[:,:,k]
					CHx1 = self.CHx1[:,:,k]
					CHx2 = self.CHx2[:,:,k]
					CHx3 = self.CHx3[:,:,k]

					diffzEy_k = (self.Ey[:,:,k] - self.Ey[:,:,k-1]) / self.dz 
					diffyEz_k = ift(self.iky * ft(self.Ez[:,:,k], axes=(1,)), axes=(1,))

					self.Bx[:,:,k] = CBx1 * self.Bx[:,:,k] + CBx2 * (diffyEz_k - diffzEy_k)
					self.Hx[:,:,k] = CHx1 * self.Hx[:,:,k] + CHx2 * self.Bx[:,:,k] + CHx3 * previous

				QBx.put(self.Bx[:,:,start:end])
				QHx.put(self.Hx[:,:,start:end])

			# Last slave node
			elif self.rank == (self.size-1):

				for k in range(start,end):

					previous = self.Bx[:,:,k].copy()

					CBx1 = self.CBx1[:,:,k]
					CBx2 = self.CBx2[:,:,k]
					CHx1 = self.CHx1[:,:,k]
					CHx2 = self.CHx2[:,:,k]
					CHx3 = self.CHx3[:,:,k]

					diffzEy_k = (self.Ey[:,:,k] - self.Ey[:,:,k-1]) / self.dz 
					diffyEz_k = ift(self.iky * ft(self.Ez[:,:,k], axes=(1,)), axes=(1,))

					self.Bx[:,:,k] = CBx1 * self.Bx[:,:,k] + CBx2 * (diffyEz_k - diffzEy_k)
					self.Hx[:,:,k] = CHx1 * self.Hx[:,:,k] + CHx2 * self.Bx[:,:,k] + CHx3 * previous

				QBx.put(self.Bx[:,:,start:end])
				QHx.put(self.Hx[:,:,start:end])

#			print("rank: {}, tstep : {}, Hx updater : {}" .format(self.rank, tstep, mp.current_process()))
			return

		def Hyupdater(start, end, QBy, QHy):

			if self.rank == 1:

				for k in range(start, end):

					previous = self.By[:,:,k].copy()

					CBy1 = self.CBy1[:,:,k]
					CBy2 = self.CBy2[:,:,k]
					CHy1 = self.CHy1[:,:,k]
					CHy2 = self.CHy2[:,:,k]
					CHy3 = self.CHy3[:,:,k]
					
					diffzEx_k = (self.Ex[:,:,k+1] - self.Ex[:,:,k]) / self.dz 
					diffxEz_k = ift((self.ikx * ft(self.Ez[:,:,k], axes=(0,))), axes=(0,))

					self.By[:,:,k] = CBy1 * self.By[:,:,k] + CBy2 * (diffzEx_k - diffxEz_k)
					self.Hy[:,:,k] = CHy1 * self.Hy[:,:,k] + CHy2 * self.By[:,:,k] + CHy3 * previous

				QBy.put(self.By[:,:,start:end])
				QHy.put(self.Hy[:,:,start:end])

			elif self.rank > 1 and self.rank < (self.size-1):

				for k in range(start,end):

					previous = self.By[:,:,k].copy()

					CBy1 = self.CBy1[:,:,k]
					CBy2 = self.CBy2[:,:,k]
					CHy1 = self.CHy1[:,:,k]
					CHy2 = self.CHy2[:,:,k]
					CHy3 = self.CHy3[:,:,k]
					
					diffzEx_k = (self.Ex[:,:,k] - self.Ex[:,:,k-1]) / self.dz 
					diffxEz_k = ift(self.ikx * ft(self.Ez[:,:,k], axes=(0,)), axes=(0,))

					self.By[:,:,k] = CBy1 * self.By[:,:,k] + CBy2 * (diffzEx_k - diffxEz_k)
					self.Hy[:,:,k] = CHy1 * self.Hy[:,:,k] + CHy2 * self.By[:,:,k] + CHy3 * previous

				QBy.put(self.By[:,:,start:end])
				QHy.put(self.Hy[:,:,start:end])

			elif self.rank == (self.size-1):

				for k in range(start,end):

					previous = self.By[:,:,k].copy()

					CBy1 = self.CBy1[:,:,k]
					CBy2 = self.CBy2[:,:,k]
					CHy1 = self.CHy1[:,:,k]
					CHy2 = self.CHy2[:,:,k]
					CHy3 = self.CHy3[:,:,k]
					
					diffzEx_k = (self.Ex[:,:,k] - self.Ex[:,:,k-1]) / self.dz 
					diffxEz_k = ift(self.ikx * ft(self.Ez[:,:,k], axes=(0,)), axes=(0,))

					self.By[:,:,k] = CBy1 * self.By[:,:,k] + CBy2 * (diffzEx_k - diffxEz_k)
					self.Hy[:,:,k] = CHy1 * self.Hy[:,:,k] + CHy2 * self.By[:,:,k] + CHy3 * previous

				QBy.put(self.By[:,:,start:end])
				QHy.put(self.Hy[:,:,start:end])

#			print("rank: {}, tstep : {}, Hy updater : {}" .format(self.rank, tstep, mp.current_process()))
			return

		def HxHyHzupdater(EEHstart, HHEstart, end, QBx, QBy, QBz, QHx, QHy, QHz):

			Hxupdater(HHEstart, end, QBx, QHx)
			Hyupdater(HHEstart, end, QBy, QHy)
			Hzupdater(EEHstart, end, QBz, QHz)

#			print("rank: {}, tstep : {}, Hxyz updater : {}" .format(self.rank, tstep, mp.current_process()))
			return

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

		#--------------------------------------------------------------#
		#------------ MPI recv Ex and Ey from next rank ---------------#
		#--------------------------------------------------------------#

		if self.rank > 0 and self.rank < (self.size-1): # rank 1,2,...,n-2

			recv_Ex_last = self.comm.recv( source=(self.rank+1), tag=(tstep*100+4))
			recv_Ey_last = self.comm.recv( source=(self.rank+1), tag=(tstep*100+5))

			self.Ex[:,:,-1] = recv_Ex_last
			self.Ey[:,:,-1] = recv_Ey_last

		if self.rank > 0:

			self.Hchildpart = 50. / 100

			# index where child part is ended and parent part is started.
			cend_pstr = np.int64(self.OWNgrid_per_node[2] * self.Hchildpart)

			if self.rank == 1:

				# Start point of child on HHE and EEH grid in the first node.
				cfHHEstr = 0
				cfEEHstr = 1

				# End point of parent on HHE and EEH grid in the first node.
				pfHHEend = self.HHEgrid_per_node[2]
				pfEEHend = self.EEHgrid_per_node[2] - 1

			elif self.rank > 1 and self.rank < (self.size-1):

				# Start point of child on HHE and EEH grid in the middle node.
				cmHHEstr = 1
				cmEEHstr = 0

				# End point of parent on HHE and EEH grid in the middle node.
				pmHHEend = self.HHEgrid_per_node[2]
				pmEEHend = self.EEHgrid_per_node[2] - 1

			elif self.rank == (self.size -1):

				# Start point of child on HHE and EEH grid in the last node.
				clHHEstr = 1
				clEEHstr = 0

				# End point of parent on HHE and EEH grid in the last node.
				plHHEend = self.HHEgrid_per_node[2] - 1
				plEEHend = self.EEHgrid_per_node[2]

			if platform.system() == 'Linux':

				mp_ctx = mp.get_context('fork')

				QBx = mp_ctx.Queue()
				QHx = mp_ctx.Queue()

				QBy = mp_ctx.Queue()
				QHy = mp_ctx.Queue()

				QBz = mp_ctx.Queue()
				QHz = mp_ctx.Queue()

				if self.rank == 1:
					#updateHx = mp_ctx.Process(target=Hxupdater, args=(cfHHEstr, cend_pstr, QBx, QHx))
					#updateHy = mp_ctx.Process(target=Hyupdater, args=(cfHHEstr, cend_pstr, QBy, QHy))
					#updateHz = mp_ctx.Process(target=Hzupdater, args=(cfEEHstr, cend_pstr, QBz, QHz))
					updateHxHyHz = mp_ctx.Process(target=HxHyHzupdater, name='HxHyHzupdater',\
									 args=(cfEEHstr, cfHHEstr, cend_pstr, QBx, QBy, QBz, QHx, QHy, QHz))

				if self.rank > 1 and self.rank < (self.size-1):
					#updateHx = mp_ctx.Process(target=Hxupdater, args=(cmHHEstr, cend_pstr, QBx, QHx))
					#updateHy = mp_ctx.Process(target=Hyupdater, args=(cmHHEstr, cend_pstr, QBy, QHy))
					#updateHz = mp_ctx.Process(target=Hzupdater, args=(cmEEHstr, cend_pstr, QBz, QHz))
					updateHxHyHz = mp_ctx.Process(target=HxHyHzupdater, name='HxHyHzupdater',\
									 args=(cmEEHstr, cmHHEstr, cend_pstr, QBx, QBy, QBz, QHx, QHy, QHz))

				if self.rank == (self.size-1):
					#updateHx = mp_ctx.Process(target=Hxupdater, args=(clHHEstr, cend_pstr, QBx, QHx))
					#updateHy = mp_ctx.Process(target=Hyupdater, args=(clHHEstr, cend_pstr, QBy, QHy))
					#updateHz = mp_ctx.Process(target=Hzupdater, args=(clEEHstr, cend_pstr, QBz, QHz))
					updateHxHyHz = mp_ctx.Process(target=HxHyHzupdater, name='HxHyHzupdater',\
									 args=(clEEHstr, clHHEstr, cend_pstr, QBx, QBy, QBz, QHx, QHy, QHz))

				updateHxHyHz.start()

			elif platform.system() == 'Windows':

				mp_ctx = mp.get_context('spawn')

				updateHx = mp_ctx.Process(target=Hxupdater, args=(0, HHEzz_sepa, QEy, QBx, QHx))
				updateHy = mp_ctx.Process(target=Hyupdater, args=(0, HHEzz_sepa, QEx, QBy, QHy))
				updateHz = mp_ctx.Process(target=Hzupdater, args=(0, EEHzz_sepa, QBz, QHz))
				updateHx.start()
				updateHy.start()
				updateHz.start()

		"""
		From here, parent process also update some part of Hx, Hy and Hz.
		Then gather field data from child processes and merge them into one piece.
		"""

		# First slave node.
		if self.rank == 1:

			# Update some part of Hx
			for k in range(cend_pstr, pfHHEend):

				previous = self.Bx[:,:,k].copy()

				CBx1 = self.CBx1[:,:,k]
				CBx2 = self.CBx2[:,:,k]
				CHx1 = self.CHx1[:,:,k]
				CHx2 = self.CHx2[:,:,k]
				CHx3 = self.CHx3[:,:,k]

				diffzEy_k = (self.Ey[:,:,k+1] - self.Ey[:,:,k]) / self.dz 
				diffyEz_k = ift(self.iky * ft(self.Ez[:,:,k], axes=(1,)), axes=(1,))

				self.Bx[:,:,k] = CBx1 * self.Bx[:,:,k] + CBx2 * (diffyEz_k - diffzEy_k)
				self.Hx[:,:,k] = CHx1 * self.Hx[:,:,k] + CHx2 * self.Bx[:,:,k] + CHx3 * previous

			# Update some part of Hy
			for k in range(cend_pstr, pfHHEend):

				previous = self.By[:,:,k].copy()

				CBy1 = self.CBy1[:,:,k]
				CBy2 = self.CBy2[:,:,k]
				CHy1 = self.CHy1[:,:,k]
				CHy2 = self.CHy2[:,:,k]
				CHy3 = self.CHy3[:,:,k]
				
				diffzEx_k = (self.Ex[:,:,k+1] - self.Ex[:,:,k]) / self.dz 
				diffxEz_k = ift((self.ikx * ft(self.Ez[:,:,k], axes=(0,))), axes=(0,))

				self.By[:,:,k] = CBy1 * self.By[:,:,k] + CBy2 * (diffzEx_k - diffxEz_k)
				self.Hy[:,:,k] = CHy1 * self.Hy[:,:,k] + CHy2 * self.By[:,:,k] + CHy3 * previous

			# Update some part of Hz
			for k in range(cend_pstr, pfEEHend):

				previous_k = self.Bz[:,:,k].copy()

				CBz1_k = self.CBz1[:,:,k]
				CBz2_k = self.CBz2[:,:,k]
				CHz1_k = self.CHz1[:,:,k]
				CHz2_k = self.CHz2[:,:,k]
				CHz3_k = self.CHz3[:,:,k]

				diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
				diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
				self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k - diffyEx_k)
				self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

#			print("Parent finished", datetime.datetime.now())

			self.Bx[:,:,cfHHEstr:cend_pstr] = QBx.get()
			self.Hx[:,:,cfHHEstr:cend_pstr] = QHx.get()

			self.By[:,:,cfHHEstr:cend_pstr] = QBy.get()
			self.Hy[:,:,cfHHEstr:cend_pstr] = QHy.get()

			self.Bz[:,:,cfEEHstr:cend_pstr] = QBz.get()
			self.Hz[:,:,cfEEHstr:cend_pstr] = QHz.get()

			# Close the Queue for Hx updater.
			QBx.close()
			QHx.close()

			# Close the Queue for Hy updater.
			QBy.close()
			QHy.close()

			# Close the Queue for Hz updater.
			QBz.close()
			QHz.close()

			# Close the Child processes.
			#updateHx.join()
			#updateHy.join()
			#updateHz.join()
			updateHxHyHz.join()

		# Middle slave nodes.
		if self.rank > 1 and self.rank < (self.size-1):

			# Update some part of Hx
			for k in range(cend_pstr, pmHHEend):

				previous = self.Bx[:,:,k].copy()

				CBx1 = self.CBx1[:,:,k]
				CBx2 = self.CBx2[:,:,k]
				CHx1 = self.CHx1[:,:,k]
				CHx2 = self.CHx2[:,:,k]
				CHx3 = self.CHx3[:,:,k]

				diffzEy_k = (self.Ey[:,:,k] - self.Ey[:,:,k-1]) / self.dz 
				diffyEz_k = ift(self.iky * ft(self.Ez[:,:,k], axes=(1,)), axes=(1,))

				self.Bx[:,:,k] = CBx1 * self.Bx[:,:,k] + CBx2 * (diffyEz_k - diffzEy_k)
				self.Hx[:,:,k] = CHx1 * self.Hx[:,:,k] + CHx2 * self.Bx[:,:,k] + CHx3 * previous

			# Update some part of Hy
			for k in range(cend_pstr, pmHHEend):

				previous = self.By[:,:,k].copy()

				CBy1 = self.CBy1[:,:,k]
				CBy2 = self.CBy2[:,:,k]
				CHy1 = self.CHy1[:,:,k]
				CHy2 = self.CHy2[:,:,k]
				CHy3 = self.CHy3[:,:,k]
				
				diffzEx_k = (self.Ex[:,:,k] - self.Ex[:,:,k-1]) / self.dz 
				diffxEz_k = ift(self.ikx * ft(self.Ez[:,:,k], axes=(0,)), axes=(0,))

				self.By[:,:,k] = CBy1 * self.By[:,:,k] + CBy2 * (diffzEx_k - diffxEz_k)
				self.Hy[:,:,k] = CHy1 * self.Hy[:,:,k] + CHy2 * self.By[:,:,k] + CHy3 * previous

			# Update some part of Hz
			for k in range(cend_pstr, pmEEHend):

				previous_k = self.Bz[:,:,k].copy()

				CBz1_k = self.CBz1[:,:,k]
				CBz2_k = self.CBz2[:,:,k]
				CHz1_k = self.CHz1[:,:,k]
				CHz2_k = self.CHz2[:,:,k]
				CHz3_k = self.CHz3[:,:,k]

				diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
				diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
				self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k- diffyEx_k)
				self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

			self.Bx[:,:,cmHHEstr:cend_pstr] = QBx.get()
			self.Hx[:,:,cmHHEstr:cend_pstr] = QHx.get()

			self.By[:,:,cmHHEstr:cend_pstr] = QBy.get()
			self.Hy[:,:,cmHHEstr:cend_pstr] = QHy.get()

			self.Bz[:,:,cmEEHstr:cend_pstr] = QBz.get()
			self.Hz[:,:,cmEEHstr:cend_pstr] = QHz.get()

			# Close the Queue for Hx updater.
			QBx.close()
			QHx.close()

			# Close the Queue for Hy updater.
			QBy.close()
			QHy.close()

			# Close the Queue for Hz updater.
			QBz.close()
			QHz.close()

			# Close the Child processes.
			#updateHx.join()
			#updateHy.join()
			#updateHz.join()
			updateHxHyHz.join()

		# Last slave node.
		if self.rank == (self.size-1):

			# Update some part of Hx
			for k in range(cend_pstr, plHHEend):

				previous = self.Bx[:,:,k].copy()

				CBx1 = self.CBx1[:,:,k]
				CBx2 = self.CBx2[:,:,k]
				CHx1 = self.CHx1[:,:,k]
				CHx2 = self.CHx2[:,:,k]
				CHx3 = self.CHx3[:,:,k]

				diffzEy_k = (self.Ey[:,:,k] - self.Ey[:,:,k-1]) / self.dz 
				diffyEz_k = ift(self.iky * ft(self.Ez[:,:,k], axes=(1,)), axes=(1,))

				self.Bx[:,:,k] = CBx1 * self.Bx[:,:,k] + CBx2 * (diffyEz_k - diffzEy_k)
				self.Hx[:,:,k] = CHx1 * self.Hx[:,:,k] + CHx2 * self.Bx[:,:,k] + CHx3 * previous

			# Update some part of Hy
			for k in range(cend_pstr, plHHEend):

				previous = self.By[:,:,k].copy()

				CBy1 = self.CBy1[:,:,k]
				CBy2 = self.CBy2[:,:,k]
				CHy1 = self.CHy1[:,:,k]
				CHy2 = self.CHy2[:,:,k]
				CHy3 = self.CHy3[:,:,k]
				
				diffzEx_k = (self.Ex[:,:,k] - self.Ex[:,:,k-1]) / self.dz 
				diffxEz_k = ift(self.ikx * ft(self.Ez[:,:,k], axes=(0,)), axes=(0,))

				self.By[:,:,k] = CBy1 * self.By[:,:,k] + CBy2 * (diffzEx_k - diffxEz_k)
				self.Hy[:,:,k] = CHy1 * self.Hy[:,:,k] + CHy2 * self.By[:,:,k] + CHy3 * previous

			# Update some part of Hz
			for k in range(cend_pstr, plEEHend):

				previous_k = self.Bz[:,:,k].copy()

				CBz1_k = self.CBz1[:,:,k]
				CBz2_k = self.CBz2[:,:,k]
				CHz1_k = self.CHz1[:,:,k]
				CHz2_k = self.CHz2[:,:,k]
				CHz3_k = self.CHz3[:,:,k]

				diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
				diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
				self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k- diffyEx_k)
				self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

			self.Bx[:,:,clHHEstr:cend_pstr] = QBx.get()
			self.Hx[:,:,clHHEstr:cend_pstr] = QHx.get()

			self.By[:,:,clHHEstr:cend_pstr] = QBy.get()
			self.Hy[:,:,clHHEstr:cend_pstr] = QHy.get()

			self.Bz[:,:,clEEHstr:cend_pstr] = QBz.get()
			self.Hz[:,:,clEEHstr:cend_pstr] = QHz.get()

			# Close the Queue for Hx updater.
			QBx.close()
			QHx.close()

			# Close the Queue for Hy updater.
			QBy.close()
			QHy.close()

			# Close the Queue for Hz updater.
			QBz.close()
			QHz.close()

			# Close the Child processes.
			#updateHx.join()
			#updateHy.join()
			#updateHz.join()
			updateHxHyHz.join()

	def updateE(self, tstep) :
		"""By defining internal function Ex updater, Ey updater and Ez updater, this method
		update the total E field which is distributed in slave nodes.

		PARAMETERS
		----------
		tstep	:	int
			time step of the simulation.

		RETURNS
		-------
		None
		"""

		def Ezupdater(start, end, QDz, QEz):

			if self.rank == 1:

				for k in range(start, end):

					previous_k = self.Dz[:,:,k].copy()

					CDz1_k = self.CDz1[:,:,k]
					CDz2_k = self.CDz2[:,:,k]
					CEz1_k = self.CEz1[:,:,k]
					CEz2_k = self.CEz2[:,:,k]
					CEz3_k = self.CEz3[:,:,k]

					diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
					diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
					self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
					self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

#				previous = self.Dz.copy()
#
#				CDz1 = self.CDz1
#				CDz2 = self.CDz2
#				CEz1 = self.CEz1
#				CEz2 = self.CEz2
#				CEz3 = self.CEz3
#
#				diffxHy = ift(self.HHEikx * ft(self.Hy, axes=(0,)), axes=(0,))
#				diffyHx = ift(self.HHEiky * ft(self.Hx, axes=(1,)), axes=(1,))
#				self.Dz = CDz1 * self.Dz + CDz2 * (diffxHy - diffyHx)
#				self.Ez = CEz1 * self.Ez + CEz2 * self.Dz + CEz3 * previous
#
				QDz.put(self.Dz[:,:,start:end])
				QEz.put(self.Ez[:,:,start:end])

			elif self.rank > 1 and self.rank < (self.size-1):

				for k in range(start, end):

					previous_k = self.Dz[:,:,k].copy()

					CDz1_k = self.CDz1[:,:,k]
					CDz2_k = self.CDz2[:,:,k]
					CEz1_k = self.CEz1[:,:,k]
					CEz2_k = self.CEz2[:,:,k]
					CEz3_k = self.CEz3[:,:,k]

					diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
					diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
					self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
					self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

#				previous = self.Dz.copy()
#
#				CDz1 = self.CDz1
#				CDz2 = self.CDz2
#				CEz1 = self.CEz1
#				CEz2 = self.CEz2
#				CEz3 = self.CEz3
#
#				diffxHy = ift(self.HHEikx * ft(self.Hy, axes=(0,)), axes=(0,))
#				diffyHx = ift(self.HHEiky * ft(self.Hx, axes=(1,)), axes=(1,))
#				self.Dz = CDz1 * self.Dz + CDz2 * (diffxHy - diffyHx)
#				self.Ez = CEz1 * self.Ez + CEz2 * self.Dz + CEz3 * previous
#
				QDz.put(self.Dz[:,:,start:end])
				QEz.put(self.Ez[:,:,start:end])

			elif self.rank == (self.size-1):

				for k in range(start, end):

					previous_k = self.Dz[:,:,k].copy()

					CDz1_k = self.CDz1[:,:,k]
					CDz2_k = self.CDz2[:,:,k]
					CEz1_k = self.CEz1[:,:,k]
					CEz2_k = self.CEz2[:,:,k]
					CEz3_k = self.CEz3[:,:,k]

					diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
					diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
					self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
					self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

#				previous = self.Dz.copy()
#
#				CDz1 = self.CDz1
#				CDz2 = self.CDz2
#				CEz1 = self.CEz1
#				CEz2 = self.CEz2
#				CEz3 = self.CEz3
#
#				diffxHy = ift(self.HHEikx * ft(self.Hy, axes=(0,)), axes=(0,))
#				diffyHx = ift(self.HHEiky * ft(self.Hx, axes=(1,)), axes=(1,))
#				self.Dz = CDz1 * self.Dz + CDz2 * (diffxHy - diffyHx)
#				self.Ez = CEz1 * self.Ez + CEz2 * self.Dz + CEz3 * previous
#
				QDz.put(self.Dz[:,:,start:end])
				QEz.put(self.Ez[:,:,start:end])

			return

		def Exupdater(start, end, QDx, QEx):

			if self.rank == 1:

				for k in range(start, end) :

					previous = self.Dx[:,:,k].copy()

					CDx1 = self.CDx1[:,:,k]
					CDx2 = self.CDx2[:,:,k]
					CEx1 = self.CEx1[:,:,k]
					CEx2 = self.CEx2[:,:,k]
					CEx3 = self.CEx3[:,:,k]
					
					diffzHy_k = (self.Hy[:,:,k] - self.Hy[:,:,k-1]) / self.dz 
					diffyHz_k = ift(self.iky * ft(self.Hz[:,:,k].copy(), axes=(1,)), axes=(1,))

					self.Dx[:,:,k] = CDx1 * self.Dx[:,:,k] + CDx2 * (diffyHz_k - diffzHy_k)
					self.Ex[:,:,k] = CEx1 * self.Ex[:,:,k] + CEx2 * self.Dx[:,:,k] + CEx3 * previous

				QDx.put(self.Dx[:,:,start:end])
				QEx.put(self.Ex[:,:,start:end])

			elif self.rank > 1 and self.rank < (self.size-1):

				for k in range(start, end):

					previous = self.Dx[:,:,k].copy()

					CDx1 = self.CDx1[:,:,k]
					CDx2 = self.CDx2[:,:,k]
					CEx1 = self.CEx1[:,:,k]
					CEx2 = self.CEx2[:,:,k]
					CEx3 = self.CEx3[:,:,k]

					diffzHy_k = (self.Hy[:,:,k+1] - self.Hy[:,:,k]) / self.dz 
					diffyHz_k = ift(self.iky * ft(self.Hz[:,:,k].copy(), axes=(1,)), axes=(1,))

					self.Dx[:,:,k] = CDx1 * self.Dx[:,:,k] + CDx2 * (diffyHz_k - diffzHy_k)
					self.Ex[:,:,k] = CEx1 * self.Ex[:,:,k] + CEx2 * self.Dx[:,:,k] + CEx3 * previous

				QDx.put(self.Dx[:,:,start:end])
				QEx.put(self.Ex[:,:,start:end])

			elif self.rank == (self.size-1):

				for k in range(start, end):

					previous = self.Dx[:,:,k].copy()

					CDx1 = self.CDx1[:,:,k]
					CDx2 = self.CDx2[:,:,k]
					CEx1 = self.CEx1[:,:,k]
					CEx2 = self.CEx2[:,:,k]
					CEx3 = self.CEx3[:,:,k]

					diffzHy_k = (self.Hy[:,:,k+1] - self.Hy[:,:,k]) / self.dz 
					diffyHz_k = ift(self.iky * ft(self.Hz[:,:,k].copy(), axes=(1,)), axes=(1,))

					self.Dx[:,:,k] = CDx1 * self.Dx[:,:,k] + CDx2 * (diffyHz_k - diffzHy_k)
					self.Ex[:,:,k] = CEx1 * self.Ex[:,:,k] + CEx2 * self.Dx[:,:,k] + CEx3 * previous

				QDx.put(self.Dx[:,:,start:end])
				QEx.put(self.Ex[:,:,start:end])

			return

		def Eyupdater(start, end, QDy, QEy):

			if self.rank == 1:

				for k in range(start, end) :

					previous = self.Dy[:,:,k].copy()

					CDy1 = self.CDy1[:,:,k]
					CDy2 = self.CDy2[:,:,k]
					CEy1 = self.CEy1[:,:,k]
					CEy2 = self.CEy2[:,:,k]
					CEy3 = self.CEy3[:,:,k]

					diffzHx_k = (self.Hx[:,:,k] - self.Hx[:,:,k-1]) / self.dz 
					diffxHz_k = ift(self.ikx * ft(self.Hz[:,:,k], axes=(0,)), axes=(0,))

					self.Dy[:,:,k] = CDy1 * self.Dy[:,:,k] + CDy2 * (diffzHx_k - diffxHz_k)
					self.Ey[:,:,k] = CEy1 * self.Ey[:,:,k] + CEy2 * self.Dy[:,:,k] + CEy3 * previous

				QDy.put(self.Dy[:,:,start:end])
				QEy.put(self.Ey[:,:,start:end])

			elif self.rank > 1 and self.rank < (self.size-1):

				for k in range(start, end):

					previous = self.Dy[:,:,k].copy()

					CDy1 = self.CDy1[:,:,k]
					CDy2 = self.CDy2[:,:,k]
					CEy1 = self.CEy1[:,:,k]
					CEy2 = self.CEy2[:,:,k]
					CEy3 = self.CEy3[:,:,k]

					diffzHx_k = (self.Hx[:,:,k+1] - self.Hx[:,:,k]) / self.dz 
					diffxHz_k = ift(self.ikx * ft(self.Hz[:,:,k], axes=(0,)), axes=(0,))

					self.Dy[:,:,k] = CDy1 * self.Dy[:,:,k] + CDy2 * (diffzHx_k - diffxHz_k)
					self.Ey[:,:,k] = CEy1 * self.Ey[:,:,k] + CEy2 * self.Dy[:,:,k] + CEy3 * previous

				QDy.put(self.Dy[:,:,start:end])
				QEy.put(self.Ey[:,:,start:end])

			elif self.rank == (self.size-1):

				for k in range(start, end):

					previous = self.Dy[:,:,k].copy()

					CDy1 = self.CDy1[:,:,k]
					CDy2 = self.CDy2[:,:,k]
					CEy1 = self.CEy1[:,:,k]
					CEy2 = self.CEy2[:,:,k]
					CEy3 = self.CEy3[:,:,k]

					diffzHx_k = (self.Hx[:,:,k+1] - self.Hx[:,:,k]) / self.dz 
					diffxHz_k = ift(self.ikx * ft(self.Hz[:,:,k], axes=(0,)), axes=(0,))

					self.Dy[:,:,k] = CDy1 * self.Dy[:,:,k] + CDy2 * (diffzHx_k - diffxHz_k)
					self.Ey[:,:,k] = CEy1 * self.Ey[:,:,k] + CEy2 * self.Dy[:,:,k] + CEy3 * previous

				QDy.put(self.Dy[:,:,start:end])
				QEy.put(self.Ey[:,:,start:end])

			return

		def ExEyEzupdater(start, EEHend, HHEend, QDx, QDy, QDz, QEx, QEy, QEz):

			Exupdater(start, EEHend, QDx, QEx)
			Eyupdater(start, EEHend, QDy, QEy)
			Ezupdater(start, HHEend, QDz, QEz)

			return

		ft  = np.fft.fftn
		ift = np.fft.ifftn
		nax = np.newaxis

		if self.rank > 0:

			self.Echildpart = 1 - self.Hchildpart

			# index where parent part is ended and child part is started.
			pend_cstr = np.int64(self.OWNgrid_per_node[2] * self.Echildpart)

			if self.rank == 1:
				# Start point of child on HHE and EEH grid in the first node.
				pfHHEstr = 0
				pfEEHstr = 1

				# End point of parent on HHE and EEH grid in the first node.
				cfHHEend = self.HHEgrid_per_node[2]
				cfEEHend = self.EEHgrid_per_node[2] - 1

			elif self.rank > 1 and self.rank < (self.size-1):
				# Start point of child on HHE and EEH grid in the middle node.
				pmHHEstr = 1
				pmEEHstr = 0

				# End point of parent on HHE and EEH grid in the middle node.
				cmHHEend = self.HHEgrid_per_node[2]
				cmEEHend = self.EEHgrid_per_node[2] - 1

			elif self.rank == (self.size-1):
				# Start point of child on HHE and EEH grid in the last node.
				plHHEstr = 1
				plEEHstr = 0

				# End point of parent on HHE and EEH grid in the last node.
				clHHEend = self.HHEgrid_per_node[2] - 1
				clEEHend = self.EEHgrid_per_node[2]

			if platform.system() == 'Linux':

				mp_ctx = mp.get_context('fork')

				QDx = mp_ctx.Queue()
				QEx = mp_ctx.Queue()

				QDy = mp_ctx.Queue()
				QEy = mp_ctx.Queue()

				QDz = mp_ctx.Queue()
				QEz = mp_ctx.Queue()

				if self.rank == 1:
					#updateEx = mp_ctx.Process(target=Exupdater, args=(pend_cstr, cfEEHend, QDx, QEx))
					#updateEy = mp_ctx.Process(target=Eyupdater, args=(pend_cstr, cfEEHend, QDy, QEy))
					#updateEz = mp_ctx.Process(target=Ezupdater, args=(pend_cstr, cfHHEend, QDz, QEz))
					updateExEyEz = mp_ctx.Process(target=ExEyEzupdater, \
									args=(pend_cstr, cfEEHend, cfHHEend, QDx, QDy, QDz, QEx, QEy, QEz))

				if self.rank > 1 and self.rank < (self.size-1):
					#updateEx = mp_ctx.Process(target=Exupdater, args=(pend_cstr, cmEEHend, QDx, QEx))
					#updateEy = mp_ctx.Process(target=Eyupdater, args=(pend_cstr, cmEEHend, QDy, QEy))
					#updateEz = mp_ctx.Process(target=Ezupdater, args=(pend_cstr, cmHHEend, QDz, QEz))
					updateExEyEz = mp_ctx.Process(target=ExEyEzupdater, \
									args=(pend_cstr, cmEEHend, cmHHEend, QDx, QDy, QDz, QEx, QEy, QEz))

				if self.rank == (self.size-1):
					#updateEx = mp_ctx.Process(target=Exupdater, args=(pend_cstr, clEEHend, QDx, QEx))
					#updateEy = mp_ctx.Process(target=Eyupdater, args=(pend_cstr, clEEHend, QDy, QEy))
					#updateEz = mp_ctx.Process(target=Ezupdater, args=(pend_cstr, clHHEend, QDz, QEz))
					updateExEyEz = mp_ctx.Process(target=ExEyEzupdater, \
									args=(pend_cstr, clEEHend, clHHEend, QDx, QDy, QDz, QEx, QEy, QEz))

				updateExEyEz.start()

			elif platform.system() == 'Windows':

				mp_ctx = mp.get_context('spawn')

				updateEx = mp_ctx.Process(target=Exupdater, args=(QHy, QDx, QEx))
				updateEy = mp_ctx.Process(target=Eyupdater, args=(QHx, QDy, QEy))
				updateEz = mp_ctx.Process(target=Ezupdater, args=(QDz,QEz))
				updateEx.start()
				updateEy.start()
				updateEz.start()

		#---------------------------------------------------------#
		#------------ MPI send Hx and Hy to next rank ------------#
		#---------------------------------------------------------#

		if self.rank > 0 and self.rank < (self.size-1): # rank 1,2,3,...,n-2

			send_Hx_last = self.Hx[:,:,-1].copy()
			send_Hy_last = self.Hy[:,:,-1].copy()

			self.comm.send(send_Hx_last, dest=(self.rank+1), tag=(tstep*100+1))
			self.comm.send(send_Hy_last, dest=(self.rank+1), tag=(tstep*100+2))

		#---------------------------------------------------------#
		#--------- MPI recv Hx and Hy from previous rank ---------#
		#---------------------------------------------------------#

		if self.rank > 1 and self.rank < self.size: # rank 2,3,...,n-1

			recv_Hx_first = self.comm.recv( source=(self.rank-1), tag=(tstep*100+1))
			recv_Hy_first = self.comm.recv( source=(self.rank-1), tag=(tstep*100+2) )

			self.Hx[:,:,0] = recv_Hx_first
			self.Hy[:,:,0] = recv_Hy_first

		# First slave node.
		if self.rank == 1:

			# Update some part of Ex
			for k in range(pfEEHstr, pend_cstr) :

				previous = self.Dx[:,:,k].copy()

				CDx1 = self.CDx1[:,:,k]
				CDx2 = self.CDx2[:,:,k]
				CEx1 = self.CEx1[:,:,k]
				CEx2 = self.CEx2[:,:,k]
				CEx3 = self.CEx3[:,:,k]
				
				diffzHy_k = (self.Hy[:,:,k] - self.Hy[:,:,k-1]) / self.dz 
				diffyHz_k = ift(self.iky * ft(self.Hz[:,:,k].copy(), axes=(1,)), axes=(1,))

				self.Dx[:,:,k] = CDx1 * self.Dx[:,:,k] + CDx2 * (diffyHz_k - diffzHy_k)
				self.Ex[:,:,k] = CEx1 * self.Ex[:,:,k] + CEx2 * self.Dx[:,:,k] + CEx3 * previous

			# Update some part of Ey
			for k in range(pfEEHstr, pend_cstr) :

				previous = self.Dy[:,:,k].copy()

				CDy1 = self.CDy1[:,:,k]
				CDy2 = self.CDy2[:,:,k]
				CEy1 = self.CEy1[:,:,k]
				CEy2 = self.CEy2[:,:,k]
				CEy3 = self.CEy3[:,:,k]

				diffzHx_k = (self.Hx[:,:,k] - self.Hx[:,:,k-1]) / self.dz 
				diffxHz_k = ift(self.ikx * ft(self.Hz[:,:,k], axes=(0,)), axes=(0,))

				self.Dy[:,:,k] = CDy1 * self.Dy[:,:,k] + CDy2 * (diffzHx_k - diffxHz_k)
				self.Ey[:,:,k] = CEy1 * self.Ey[:,:,k] + CEy2 * self.Dy[:,:,k] + CEy3 * previous

			# Update some part of Ez
			for k in range(pfHHEstr, pend_cstr):

				previous_k = self.Dz[:,:,k].copy()

				CDz1_k = self.CDz1[:,:,k]
				CDz2_k = self.CDz2[:,:,k]
				CEz1_k = self.CEz1[:,:,k]
				CEz2_k = self.CEz2[:,:,k]
				CEz3_k = self.CEz3[:,:,k]

				diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
				diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
				self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
				self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

			self.Dx[:,:,pend_cstr:cfEEHend] = QDx.get()
			self.Ex[:,:,pend_cstr:cfEEHend] = QEx.get()

			self.Dy[:,:,pend_cstr:cfEEHend] = QDy.get()
			self.Ey[:,:,pend_cstr:cfEEHend] = QEy.get()

			self.Dz[:,:,pend_cstr:cfHHEend] = QDz.get()
			self.Ez[:,:,pend_cstr:cfHHEend] = QEz.get()

			# Close the Queue for Ex updater.
			QDx.close()
			QEx.close()

			# Close the Queue for Ey updater.
			QDy.close()
			QEy.close()

			# Close the Queue for Ez updater.
			QDz.close()
			QEz.close()

			# Close the Child processes.
			#updateEx.join()
			#updateEy.join()
			#updateEz.join()
			updateExEyEz.join()

		# Middle slave nodes.
		if self.rank > 1 and self.rank < (self.size-1):

			# Update some part of Ex.
			for k in range(pmEEHstr, pend_cstr):

				previous = self.Dx[:,:,k].copy()

				CDx1 = self.CDx1[:,:,k]
				CDx2 = self.CDx2[:,:,k]
				CEx1 = self.CEx1[:,:,k]
				CEx2 = self.CEx2[:,:,k]
				CEx3 = self.CEx3[:,:,k]

				diffzHy_k = (self.Hy[:,:,k+1] - self.Hy[:,:,k]) / self.dz 
				diffyHz_k = ift(self.iky * ft(self.Hz[:,:,k].copy(), axes=(1,)), axes=(1,))

				self.Dx[:,:,k] = CDx1 * self.Dx[:,:,k] + CDx2 * (diffyHz_k - diffzHy_k)
				self.Ex[:,:,k] = CEx1 * self.Ex[:,:,k] + CEx2 * self.Dx[:,:,k] + CEx3 * previous

			# Update some part of Ey.
			for k in range(pmEEHstr, pend_cstr):

				previous = self.Dy[:,:,k].copy()

				CDy1 = self.CDy1[:,:,k]
				CDy2 = self.CDy2[:,:,k]
				CEy1 = self.CEy1[:,:,k]
				CEy2 = self.CEy2[:,:,k]
				CEy3 = self.CEy3[:,:,k]

				diffzHx_k = (self.Hx[:,:,k+1] - self.Hx[:,:,k]) / self.dz 
				diffxHz_k = ift(self.ikx * ft(self.Hz[:,:,k], axes=(0,)), axes=(0,))

				self.Dy[:,:,k] = CDy1 * self.Dy[:,:,k] + CDy2 * (diffzHx_k - diffxHz_k)
				self.Ey[:,:,k] = CEy1 * self.Ey[:,:,k] + CEy2 * self.Dy[:,:,k] + CEy3 * previous

			# Update some part of Ez.
			for k in range(pmHHEstr, pend_cstr):

				previous_k = self.Dz[:,:,k].copy()

				CDz1_k = self.CDz1[:,:,k]
				CDz2_k = self.CDz2[:,:,k]
				CEz1_k = self.CEz1[:,:,k]
				CEz2_k = self.CEz2[:,:,k]
				CEz3_k = self.CEz3[:,:,k]

				diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
				diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
				self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
				self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

			self.Dx[:,:,pend_cstr:cmEEHend] = QDx.get()
			self.Ex[:,:,pend_cstr:cmEEHend] = QEx.get()

			self.Dy[:,:,pend_cstr:cmEEHend] = QDy.get()
			self.Ey[:,:,pend_cstr:cmEEHend] = QEy.get()

			self.Dz[:,:,pend_cstr:cmHHEend] = QDz.get()
			self.Ez[:,:,pend_cstr:cmHHEend] = QEz.get()

			# Close the Queue for Ex updater.
			QDx.close()
			QEx.close()

			# Close the Queue for Ey updater.
			QDy.close()
			QEy.close()

			# Close the Queue for Ez updater.
			QDz.close()
			QEz.close()

			# Close the Child processes.
			#updateEx.join()
			#updateEy.join()
			#updateEz.join()
			updateExEyEz.join()

		# Last slave node
		if self.rank == (self.size-1):

			# Update some part of Ex.
			for k in range(plEEHstr, pend_cstr):

				previous = self.Dx[:,:,k].copy()

				CDx1 = self.CDx1[:,:,k]
				CDx2 = self.CDx2[:,:,k]
				CEx1 = self.CEx1[:,:,k]
				CEx2 = self.CEx2[:,:,k]
				CEx3 = self.CEx3[:,:,k]

				diffzHy_k = (self.Hy[:,:,k+1] - self.Hy[:,:,k]) / self.dz 
				diffyHz_k = ift(self.iky * ft(self.Hz[:,:,k].copy(), axes=(1,)), axes=(1,))

				self.Dx[:,:,k] = CDx1 * self.Dx[:,:,k] + CDx2 * (diffyHz_k - diffzHy_k)
				self.Ex[:,:,k] = CEx1 * self.Ex[:,:,k] + CEx2 * self.Dx[:,:,k] + CEx3 * previous

			# Update some part of Ey.
			for k in range(plEEHstr, pend_cstr):

				previous = self.Dy[:,:,k].copy()

				CDy1 = self.CDy1[:,:,k]
				CDy2 = self.CDy2[:,:,k]
				CEy1 = self.CEy1[:,:,k]
				CEy2 = self.CEy2[:,:,k]
				CEy3 = self.CEy3[:,:,k]

				diffzHx_k = (self.Hx[:,:,k+1] - self.Hx[:,:,k]) / self.dz 
				diffxHz_k = ift(self.ikx * ft(self.Hz[:,:,k], axes=(0,)), axes=(0,))

				self.Dy[:,:,k] = CDy1 * self.Dy[:,:,k] + CDy2 * (diffzHx_k - diffxHz_k)
				self.Ey[:,:,k] = CEy1 * self.Ey[:,:,k] + CEy2 * self.Dy[:,:,k] + CEy3 * previous

			# Update some part of Ez.
			for k in range(plHHEstr, pend_cstr):

				previous_k = self.Dz[:,:,k].copy()

				CDz1_k = self.CDz1[:,:,k]
				CDz2_k = self.CDz2[:,:,k]
				CEz1_k = self.CEz1[:,:,k]
				CEz2_k = self.CEz2[:,:,k]
				CEz3_k = self.CEz3[:,:,k]

				diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
				diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
				self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
				self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

			self.Dx[:,:,pend_cstr:clEEHend] = QDx.get()
			self.Ex[:,:,pend_cstr:clEEHend] = QEx.get()

			self.Dy[:,:,pend_cstr:clEEHend] = QDy.get()
			self.Ey[:,:,pend_cstr:clEEHend] = QEy.get()

			self.Dz[:,:,pend_cstr:clHHEend] = QDz.get()
			self.Ez[:,:,pend_cstr:clHHEend] = QEz.get()

			# Close the Queue for Ex updater.
			QDx.close()
			QEx.close()

			# Close the Queue for Ey updater.
			QDy.close()
			QEy.close()

			# Close the Queue for Ez updater.
			QDz.close()
			QEz.close()

			# Close the Child processes.
			#updateEx.join()
			#updateEy.join()
			#updateEz.join()
			updateExEyEz.join()

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
