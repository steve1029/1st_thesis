import numpy as np
from scipy.constants import c, mu_0, epsilon_0
from build import Space, Fields

class Material(object):

	def __init__(self,Space):
		
		self.rank = Space.rank
		self.comm = Space.comm

		if Space.rank == 0 :

			self.gridx = Space.gridx
			self.gridy = Space.gridy
			self.gridz = Space.gridz

			self.space_eps_on  = Space.space_eps_on
			self.space_eps_off = Space.space_eps_off
			self.space_mu_on   = Space.space_mu_on
			self.space_mu_off  = Space.space_mu_off

			self.Esigma_onx = Space.Esigma_onx
			self.Esigma_ony = Space.Esigma_ony
			self.Esigma_onz = Space.Esigma_onz

			self.Esigma_offx = Space.Esigma_offx
			self.Esigma_offy = Space.Esigma_offy
			self.Esigma_offz = Space.Esigma_offz

		else : pass

class Box(Material):
	"""Set a rectangular box on a simulation space.
	
	PARAMETERS
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
	None
	"""
	
	def __init__(self, Space, start, end, eps_r, mu_r, sigma):

		if Space.rank == 0 :

			Material.__init__(self, Space)

			assert len(start)  == 3, "Only 3D material is possible."
			assert len(end  )  == 3, "Only 3D material is possible."

			if type(eps_r)=='list' or type(eps_r) == 'tuple':
				assert len(eps_r) != 2, "eps_r is a number or a list(tuple) with len 3."	
			if type(mu_r)=='list' or type(mu_r ) == 'tuple':
				assert len(mu_r)  != 2, "eps_r is a number or a list(tuple) with len 3."	
			if type(sigma)=='list' or type(sigma) == 'tuple': 
				assert len(sigma) != 2, "eps_r is a number or a list(tuple) with len 3."	

				if len(sigma) == 3:
				
					sigma_x = sigma[0]
					sigma_y = sigma[1]
					sigma_z = sigma[2]
			
			else: sigma_x = sigma_y = sigma_z = sigma

			height        = slice(start[0],end[0] )
			width         = slice(start[1],end[1] )
			thickness_on  = slice(start[2],end[2] )
			thickness_off = slice(start[2],end[2]-1)

			self.space_eps_on  [height,width,thickness_on ] *= eps_r
			self.space_eps_off [height,width,thickness_off] *= eps_r
			self.space_mu_on   [height,width,thickness_on ] *= mu_r
			self.space_mu_off  [height,width,thickness_off] *= mu_r

			self.Esigma_onx  [height,width,thickness_on ] = sigma_x
			self.Esigma_ony  [height,width,thickness_on ] = sigma_y
			self.Esigma_onz  [height,width,thickness_on ] = sigma_z
			self.Esigma_offx [height,width,thickness_off] = sigma_x
			self.Esigma_offy [height,width,thickness_off] = sigma_y
			self.Esigma_offz [height,width,thickness_off] = sigma_z

			#print(self.space_eps_on[height,width,thickness_on].shape)
			#print(self.space_eps_on[height,width,thickness_on]/epsilon_0)
		else : pass

		Space.comm.Barrier()

		return

class Sphere(Material):

	def __init__(self, Space, center, radius, eps_r, mu_r, sigma):

		if Space.rank == 0:

			Material.__init__(self, Space)

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
