import numpy as np
from scipy.constants import c, mu_0, epsilon_0

class Structure(object):
	def __init__(self, Space):
		self.Space = Space
		pass


class Box(Structure):
	"""Set a rectangular box on a simulation space.
	
	Parameters
	----------

	eps_r : float
			Relative electric constant or permitivity.

	mu_ r : float
			Relative magnetic constant or permeability.
		
	sigma : float
			conductivity of the material.

	size  : a list or tuple (iterable object) of ints
			x: height, y: width, z: thickness of a box.

	loc   : a list or typle (iterable objext) of ints
			x : x coordinate of bottom left upper coner
			y : y coordinate of bottom left upper coner
			z : z coordinate of bottom left upper coner

	Returns
	-------
	message	: string
		each node speaks the local location of structure.
	"""
	
	def __init__(self, Space, srt, end):
		"""Define the location of Box in each node
		
		Parameters
		-----------------------
		srt	: array_like
			A coordinate of a vertex at the upper left of the bottom
		end	: array_like
			A coordinate of a vertex at the lower right of the bottom

		Returns
		-----------------------
		None
		"""

		Structure.__init__(self, Space)

		xsrt = srt[0]
		ysrt = srt[1]
		zsrt = srt[2]

		xend = end[0]
		yend = end[1]
		zend = end[2]

		assert xsrt < xend
		assert ysrt < yend
		assert zsrt < zend

		MPIrank = self.Space.MPIrank
		MPIsize = self.Space.MPIsize

		# global X index of each node
		gxsrt = self.Space.myNx_indice[MPIrank][0]
		gxend = self.Space.myNx_indice[MPIrank][1]

		if xend <  gxsrt:
			self.global_loc = None
			self. local_loc = None
		if xsrt <  gxsrt and xend >= gxsrt and xend <= gxend:
			self.global_loc = ((gxsrt      ,ysrt,zsrt), ( xend      ,yend,zend))
			self. local_loc = ((gxsrt-gxsrt,ysrt,zsrt), ( xend-gxsrt,yend,zend))
		if xsrt <  gxsrt and xend > gxend:
			self.global_loc = ((gxsrt      ,ysrt,zsrt), (gxend      ,yend,zend))
			self. local_loc = ((gxsrt-gxsrt,ysrt,zsrt), (gxend-gxsrt,yend,zend))
		if xsrt >= gxsrt and xsrt <  gxend and xend <  gxend:
			self.global_loc = (( xsrt      ,ysrt,zsrt), ( xend      ,yend,zend))
			self. local_loc = (( xsrt-gxsrt,ysrt,zsrt), ( xend-gxsrt,yend,zend))
		if xsrt >= gxsrt and xsrt <  gxend and xend >= gxend:
			self.global_loc = (( xsrt      ,ysrt,zsrt), (gxend      ,yend,zend))
			self. local_loc = (( xsrt-gxsrt,ysrt,zsrt), (gxend-gxsrt,yend,zend))
		if xsrt >= gxend:
			self.global_loc = None
			self. local_loc = None
		
		if self.global_loc != None:
			self.local_size = (self.local_loc[1][0] - self.local_loc[0][0], yend-ysrt, zend-zsrt)
			print("rank {:>2}: x idx of structure >>> global \"{:4d},{:4d}\" and local \"{:4d},{:4d}\"" \
					.format(MPIrank, self.global_loc[0][0], self.global_loc[1][0], self.local_loc[0][0], self.local_loc[1][0]))

		self.Space.comm.Barrier()


class Sphere(Structure):

	def __init__(self, Space, center, radius, eps_r, mu_r, sigma):

		if Space.rank == 0:

			Structure.__init__(self, Space)

			x = center[0]
			y = center[1]
			z = center[2]

			for k in range(self.gridz):
				for j in range(self.gridy):
					for i in range(self.gridx):
						if ((i-x)**2 + (j-y)**2 + (k-z)**2) < (radius**2):

							self.space_eps_on[i,j,k] *= eps_r
							self.space_mu_on [i,j,k] *= mu_r

							self.Esigma_onx[i,j,k] = sigma
							self.Esigma_ony[i,j,k] = sigma
							self.Esigma_onz[i,j,k] = sigma

							self.space_eps_off[i,j,k] *= eps_r
							self.space_mu_off [i,j,k] *= mu_r

							self.Esigma_offx[i,j,k] = sigma
							self.Esigma_offy[i,j,k] = sigma
							self.Esigma_offz[i,j,k] = sigma

		else: pass

		Space.comm.Barrier()

		return
