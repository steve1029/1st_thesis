import numpy as np
import os, platform
from scipy.constants import c, mu_0, epsilon_0

class UPML(object):
	
	def __init__(self, Space,region,npml):
		"""Spacify the parameters for UPML.

		Parameters
		-----------
		Space	:	structure.Space object
			Space object specifies the grid, gridgap, courant number, etc.

		region	:	dictionary
			Set PML region as dictionary form.
			'+'  means PML regions of righthand side of the grid and
			'-'  means PML regions of lefthand side of the grid.
			'+-' is the combination of '+' and '-'.

		npml : int
			The number of grid point of the PML region.
		"""
		self.rank = Space.rank
		self.comm = Space.comm
			
		self.comm.Barrier()

		if self.rank == 0 : # only rank 0 calculate coeff array

			self.dtype = Space.dtype
			self.dt    = Space.dt
			self.npml  = npml

			IEp = Space.gridx
			JEp = Space.gridy
			KEp = Space.gridz
			
			dx  = Space.dx
			dy  = Space.dy
			dz  = Space.dz

			self.kappa_onx = Space.kappa_onx
			self.kappa_ony = Space.kappa_ony
			self.kappa_onz = Space.kappa_onz

			self.kappa_offx = Space.kappa_offx
			self.kappa_offy = Space.kappa_offy
			self.kappa_offz = Space.kappa_offz

			self.Esigma_onx = Space.Esigma_onx
			self.Esigma_ony = Space.Esigma_ony
			self.Esigma_onz = Space.Esigma_onz

			self.Hsigma_onx = Space.Hsigma_onx
			self.Hsigma_ony = Space.Hsigma_ony
			self.Hsigma_onz = Space.Hsigma_onz

			self.Esigma_offx = Space.Esigma_offx
			self.Esigma_offy = Space.Esigma_offy
			self.Esigma_offz = Space.Esigma_offz

			self.Hsigma_offx = Space.Hsigma_offx
			self.Hsigma_offy = Space.Hsigma_offy
			self.Hsigma_offz = Space.Hsigma_offz

			#----------------------------------------------------------------------------------#
			#----------------------------- Grading of PML region ------------------------------#
			#----------------------------------------------------------------------------------#

			self.rc0   = 1.e-16						# reflection coefficient
			self.imp   = np.sqrt(mu_0/epsilon_0)	# impedence
			self.gO    = 3.							# gradingOrder
			self.bdw_x = self.npml * dx				# PML thickness along x (Boundarywidth)
			self.bdw_y = self.npml * dy				# PML thickness along y
			self.bdw_z = self.npml * dz				# PML thickness along z

			self.Emaxsig_onx = -(self.gO+1) * np.log(self.rc0) / (2*self.imp*self.bdw_x)
			self.Emaxsig_ony = -(self.gO+1) * np.log(self.rc0) / (2*self.imp*self.bdw_y)
			self.Emaxsig_onz = -(self.gO+1) * np.log(self.rc0) / (2*self.imp*self.bdw_z)

			#self.Esigmax_onx = self.Emaxsig_onx * self.imp**2
			#self.Esigmax_ony = self.Emaxsig_ony * self.imp**2
			#self.Esigmax_onz = self.Emaxsig_onz * self.imp**2

			self.maxkap_onx = 5.
			self.maxkap_ony = 5.
			self.maxkap_onz = 5.

			self.region = region

			for key, value in list(self.region.items()) :
				
				if key == 'x':

					for i in range(self.npml):

						if value == '-':

							on_grid    = np.float64((self.npml-i   )) / self.npml
							#off_grid_m = np.float64((self.npml-i-.5)) / self.npml

							self.Esigma_onx[i,:,:] = self.Emaxsig_onx * (on_grid**self.gO)
							self.Hsigma_onx[i,:,:] = self.Emaxsig_onx * (on_grid**self.gO) * (self.imp**2)
							self. kappa_onx[i,:,:] = 1 + ((self.maxkap_onx-1) * (on_grid**self.gO))

							#self.Esigma_offx[i,:,:] = self.Emaxsig_onx * (off_grid_m**self.gO)
							#self.Hsigma_offx[i,:,:] = self.Emaxsig_onx * (off_grid_m**self.gO) * (self.imp**2)
							#self. kappa_offx[i,:,:] = 1 + ((self.maxkap_onx-1) * (off_grid_m**self.gO))

						elif value == '+' :

							on_grid = np.float64((self.npml-i))/self.npml
							#off_grid_p = np.float64((self.npml-i+.5)) / self.npml

							self.Esigma_onx[-i-1,:,:] = self.Emaxsig_onx * (on_grid**self.gO)
							self.Hsigma_onx[-i-1,:,:] = self.Emaxsig_onx * (on_grid**self.gO) * (self.imp**2)
							self. kappa_onx[-i-1,:,:] = 1 + ((self.maxkap_onx-1) * (on_grid**self.gO))

							#self.Esigma_offx[-i-1,:,:] = self.maxsig_Ex * (off_grid_p**self.gO)
							#self.Hsigma_offx[-i-1,:,:] = self.maxsig_Ex * (off_grid_p**self.gO) * (self.imp**2)
							#self. kappa_offx[-i-1,:,:] = 1 + ((self.maxkap_Ex-1) * (off_grid_p**self.gO))

						elif value == '+-' or value =='-+' :

							on_grid = np.float64((self.npml-i))/self.npml
							#off_grid_m = np.float64((self.npml-i-.5)) / self.npml
							#off_grid_p = np.float64((self.npml-i+.5)) / self.npml

							self.Esigma_onx[i,:,:] = self.Emaxsig_onx * (on_grid**self.gO)
							self.Hsigma_onx[i,:,:] = self.Emaxsig_onx * (on_grid**self.gO) * (self.imp**2)
							self. kappa_onx[i,:,:] = 1 + ((self.maxkap_onx-1) * (on_grid**self.gO))

							self.Esigma_onx[-i-1,:,:] = self.Esigma_onx[i,:,:]
							self.Hsigma_onx[-i-1,:,:] = self.Hsigma_onx[i,:,:]
							self. kappa_onx[-i-1,:,:] = self. kappa_onx[i,:,:]

							#self.Esigma_offx[i,:,:] = self.maxsig_Ex * (off_grid_m**self.gO)
							#self.Hsigma_offx[i,:,:] = self.maxsig_Ex * (off_grid_m**self.gO) * (self.imp**2)
							#self. kappa_offx[i,:,:] = 1 + ((self.maxkap_Ex-1) * (off_grid_m**self.gO))

							#self.Esigma_offx[-i-1,:,:] = self.maxsig_Ex * (off_grid_p**self.gO)
							#self.Hsigma_offx[-i-1,:,:] = self.maxsig_Ex * (off_grid_p**self.gO) * (self.imp**2)
							#self. kappa_offx[-i-1,:,:] = 1 + ((self.maxkap_Ex-1) * (off_grid_p**self.gO))
					
				elif key == 'y':

					for j in range(self.npml):

						if value == '-':

							on_grid    = np.float64((self.npml-j   ))/self.npml
							#off_grid_m = np.float64((self.npml-j-.5))/self.npml

							self.Esigma_ony[:,j,:] = self.Emaxsig_ony * (on_grid**self.gO)
							self.Hsigma_ony[:,j,:] = self.Emaxsig_ony * (on_grid**self.gO) * (self.imp**2)
							self. kappa_ony[:,j,:] = 1 + ((self.maxkap_ony-1) * (on_grid**self.gO))

							#self.Esigma_offy[:,j,:] = self.Emaxsig_ony * (off_grid_m**self.gO)
							#self.Hsigma_offy[:,j,:] = self.Emaxsig_ony * (off_grid_m**self.gO) * (self.imp**2)
							#self. kappa_offy[:,j,:] = 1 + ((self.maxkap_ony-1) * (off_grid_m**self.gO))

						elif value == '+' :

							on_grid    = np.float64((self.npml-j   )) / self.npml
							#off_grid_p = np.float64((self.npml-j+.5)) / self.npml

							self.Esigma_ony[:,-j-1,:] = self.Emaxsig_ony * (on_grid**self.gO)
							self.Hsigma_ony[:,-j-1,:] = self.Emaxsig_ony * (on_grid**self.gO) * (self.imp**2)
							self. kappa_ony[:,-j-1,:] = 1 + ((self.maxkap_ony-1) * (on_grid**self.gO))

							#self.Esigma_offy[:,-j-1,:] = self.Emaxsig_ony * (off_grid_p**self.gO)
							#self.Hsigma_offy[:,-j-1,:] = self.Emaxsig_ony * (off_grid_p**self.gO) * (self.imp**2)
							#self. kappa_offy[:,-j-1,:] = 1 + ((self.maxkap_ony-1) * (off_grid_p**self.gO))

						elif value == '+-' or value =='-+' :

							on_grid    = np.float64((self.npml-j  )) / self.npml
							#off_grid_m = np.float64((self.npml-j-.5)) / self.npml
							#off_grid_p = np.float64((self.npml-j+.5)) / self.npml

							self.Esigma_ony[:,j,:] = self.Emaxsig_ony * (on_grid**self.gO)
							self.Hsigma_ony[:,j,:] = self.Emaxsig_ony * (on_grid**self.gO) * (self.imp**2)
							self. kappa_ony[:,j,:] = 1 + ((self.maxkap_ony-1) * (on_grid**self.gO))

							self.Esigma_ony[:,-j-1,:] = self.Emaxsig_ony * (on_grid**self.gO)
							self.Hsigma_ony[:,-j-1,:] = self.Emaxsig_ony * (on_grid**self.gO) * (self.imp**2)
							self. kappa_ony[:,-j-1,:] = 1 + ((self.maxkap_ony-1) * (on_grid**self.gO))

							#self.Esigma_offy[:,j,:] = self.Emaxsig_ony * (off_grid_m**self.gO)
							#self.Hsigma_offy[:,j,:] = self.Emaxsig_ony * (off_grid_m**self.gO) * (self.imp**2)
							#self. kappa_offy[:,j,:] = 1 + ((self.maxkap_ony-1) * (off_grid_m**self.gO))

							#self.Esigma_offy[:,-j-1,:] = self.Emaxsig_ony * (off_grid_p**self.gO)
							#self.Hsigma_offy[:,-j-1,:] = self.Emaxsig_ony * (off_grid_p**self.gO) * (self.imp**2)
							#self. kappa_offy[:,-j-1,:] = 1 + ((self.maxkap_ony-1) * (off_grid_p**self.gO))

				elif key == 'z':

					for k in range(self.npml):

						if value == '-':

							on_grid    = np.float64((self.npml-k   )) / self.npml
							off_grid_m = np.float64((self.npml-k-.5)) / self.npml

							self.Esigma_onz[:,:,k] = self.Emaxsig_onz * (on_grid**self.gO)
							self.Hsigma_onz[:,:,k] = self.Emaxsig_onz * (on_grid**self.gO) * (self.imp**2)
							self. kappa_onz[:,:,k] = 1 + ((self.maxkap_onz-1) * (on_grid**self.gO))

							self.Esigma_offz[:,:,k] = self.Emaxsig_onz * (off_grid_m**self.gO)
							self.Hsigma_offz[:,:,k] = self.Emaxsig_onz * (off_grid_m**self.gO) * (self.imp**2)
							self. kappa_offz[:,:,k] = 1 + ((self.maxkap_onz-1) * (off_grid_m**self.gO))

						elif value == '+' :

							on_grid    = np.float64((self.npml-k   )) / self.npml
							off_grid_p = np.float64((self.npml-k+.5)) / self.npml

							self.Esigma_onz[:,:,-k-1] = self.Emaxsig_onz * (on_grid**self.gO)
							self.Hsigma_onz[:,:,-k-1] = self.Emaxsig_onz * (on_grid**self.gO) * (self.imp**2)
							self. kappa_onz[:,:,-k-1] = 1 + ((self.maxkap_onz-1) * (on_grid**self.gO))

							self.Esigma_offz[:,:,-k-1] = self.Emaxsig_onz * (off_grid_p**self.gO)
							self.Hsigma_offz[:,:,-k-1] = self.Emaxsig_onz * (off_grid_p**self.gO) * (self.imp**2)
							self. kappa_offz[:,:,-k-1] = 1 + ((self.maxkap_onz-1) * (off_grid_p**self.gO))

						elif value == '+-' or value =='-+' :

							on_grid    = np.float64((self.npml-k   )) / self.npml
							off_grid_m = np.float64((self.npml-k-.5)) / self.npml
							off_grid_p = np.float64((self.npml-k+.5)) / self.npml

							self.Esigma_onz[:,:,k] = self.Emaxsig_onz * (on_grid**self.gO)
							self.Hsigma_onz[:,:,k] = self.Emaxsig_onz * (on_grid**self.gO) * (self.imp**2)
							self. kappa_onz[:,:,k] = 1 + ((self.maxkap_onz-1) * (on_grid**self.gO))

							self.Esigma_onz[:,:,-k-1] = self.Esigma_onz[:,:,k]
							self.Hsigma_onz[:,:,-k-1] = self.Hsigma_onz[:,:,k]
							self. kappa_onz[:,:,-k-1] = self. kappa_onz[:,:,k]

							self.Esigma_offz[:,:,k] = self.Emaxsig_onz * (off_grid_m**self.gO)
							self.Hsigma_offz[:,:,k] = self.Emaxsig_onz * (off_grid_m**self.gO) * (self.imp**2)
							self. kappa_offz[:,:,k] = 1 + ((self.maxkap_onz-1) * (off_grid_m**self.gO))

							self.Esigma_offz[:,:,-k-1] = self.Emaxsig_onz * (off_grid_p**self.gO)
							self.Hsigma_offz[:,:,-k-1] = self.Emaxsig_onz * (off_grid_p**self.gO) * (self.imp**2)
							self. kappa_offz[:,:,-k-1] = 1 + ((self.maxkap_onz-1) * (off_grid_p**self.gO))

			self.onEpx = (2 * epsilon_0 * self.kappa_onx) + (self.Esigma_onx * self.dt)
			self.onEpy = (2 * epsilon_0 * self.kappa_ony) + (self.Esigma_ony * self.dt)
			self.onEpz = (2 * epsilon_0 * self.kappa_onz) + (self.Esigma_onz * self.dt)
			
			self.onEmx = (2 * epsilon_0 * self.kappa_onx) - (self.Esigma_onx * self.dt)
			self.onEmy = (2 * epsilon_0 * self.kappa_ony) - (self.Esigma_ony * self.dt)
			self.onEmz = (2 * epsilon_0 * self.kappa_onz) - (self.Esigma_onz * self.dt)

			self.onHpx = (2 * mu_0 * self.kappa_onx) + (self.Hsigma_onx * self.dt)
			self.onHpy = (2 * mu_0 * self.kappa_ony) + (self.Hsigma_ony * self.dt)
			self.onHpz = (2 * mu_0 * self.kappa_onz) + (self.Hsigma_onz * self.dt)
			
			self.onHmx = (2 * mu_0 * self.kappa_onx) - (self.Hsigma_onx * self.dt)
			self.onHmy = (2 * mu_0 * self.kappa_ony) - (self.Hsigma_ony * self.dt)
			self.onHmz = (2 * mu_0 * self.kappa_onz) - (self.Hsigma_onz * self.dt)

			self.offEpx = (2 * epsilon_0 * self.kappa_offx) + (self.Esigma_offx * self.dt)
			self.offEpy = (2 * epsilon_0 * self.kappa_offy) + (self.Esigma_offy * self.dt)
			self.offEpz = (2 * epsilon_0 * self.kappa_offz) + (self.Esigma_offz * self.dt)
			
			self.offEmx = (2 * epsilon_0 * self.kappa_offx) - (self.Esigma_offx * self.dt)
			self.offEmy = (2 * epsilon_0 * self.kappa_offy) - (self.Esigma_offy * self.dt)
			self.offEmz = (2 * epsilon_0 * self.kappa_offz) - (self.Esigma_offz * self.dt)

			self.offHpx = (2 * mu_0 * self.kappa_offx) + (self.Hsigma_offx * self.dt)
			self.offHpy = (2 * mu_0 * self.kappa_offy) + (self.Hsigma_offy * self.dt)
			self.offHpz = (2 * mu_0 * self.kappa_offz) + (self.Hsigma_offz * self.dt)
			
			self.offHmx = (2 * mu_0 * self.kappa_offx) - (self.Hsigma_offx * self.dt)
			self.offHmy = (2 * mu_0 * self.kappa_offy) - (self.Hsigma_offy * self.dt)
			self.offHmz = (2 * mu_0 * self.kappa_offz) - (self.Hsigma_offz * self.dt)
			
			dt = self.dt
			nax = np.newaxis

			self.space_eps_on  = Space.space_eps_on
			self.space_eps_off = Space.space_eps_off
			self.space_mu_on   = Space.space_mu_on
			self.space_mu_off  = Space.space_mu_off

			self.CDx1 = (self.onEmy / self.onEpy)
			self.CDy1 = (self.onEmz / self.onEpz)
			self.CDz1 = (self.onEmx / self.onEpx)

			self.CDx2 = 2. * epsilon_0 * dt / self.onEpy
			self.CDy2 = 2. * epsilon_0 * dt / self.onEpz
			self.CDz2 = 2. * epsilon_0 * dt / self.onEpx

			self.CEx1 = self.onEmz / self.onEpz
			self.CEx2 = self.onEpx / self.onEpz / self.space_eps_on
			self.CEx3 = self.onEmx / self.onEpz / self.space_eps_on * (-1)

			self.CEy1 = self.onEmx / self.onEpx
			self.CEy2 = self.onEpy / self.onEpx / self.space_eps_on
			self.CEy3 = self.onEmy / self.onEpx / self.space_eps_on * (-1)

			self.CEz1 = self.onEmy  / self.onEpy
			self.CEz2 = self.offEpz / self.onEpy / self.space_eps_off
			self.CEz3 = self.offEmz / self.onEpy / self.space_eps_off * (-1)

			self.CBx1 = self.onHmy  / self. onHpy
			self.CBy1 = self.offHmz / self.offHpz
			self.CBz1 = self.onHmx  / self. onHpx

			self.CBx2 = 2. * mu_0 * dt / self. onHpy * (-1)
			self.CBy2 = 2. * mu_0 * dt / self.offHpz * (-1)
			self.CBz2 = 2. * mu_0 * dt / self. onHpx * (-1)

			self.CHx1 = self.offHmz / self.offHpz
			self.CHx2 = self. onHpx / self.offHpz / self.space_mu_off
			self.CHx3 = self. onHmx / self.offHpz / self.space_mu_off * (-1)

			self.CHy1 = self.onHmx / self.onHpx
			self.CHy2 = self.onHpy / self.onHpx / self.space_mu_off
			self.CHy3 = self.onHmy / self.onHpx / self.space_mu_off * (-1)

			self.CHz1 = self.onHmy / self.onHpy
			self.CHz2 = self.onHpz / self.onHpy / self.space_mu_on
			self.CHz3 = self.onHmz / self.onHpy / self.space_mu_on * (-1)

			self.coefflist = [self.CDx1, self.CDx2, self.CBx1, self.CBx2, \
							  self.CDy1, self.CDy2, self.CBy1, self.CBy2, \
							  self.CDz1, self.CDz2, self.CBz1, self.CBz2, \

							  self.CEx1, self.CEx2, self.CEx3, self.CHx1, self.CHx2, self.CHx3, \
							  self.CEy1, self.CEy2, self.CEy3, self.CHy1, self.CHy2, self.CHy3, \
							  self.CEz1, self.CEz2, self.CEz3, self.CHz1, self.CHz2, self.CHz3  ]
		
			for num, item in enumerate(self.coefflist) : assert np.all(item) == True

		else : pass

		self.comm.Barrier()
	
	def save_coeff_data(self,path) :

		self.comm.Barrier()
		
		if self.rank == 0 :

			if not os.path.exists(path): os.mkdir(path)

			#if   platform.system() == 'Windows' : nl = '\r\n'
			#elif platform.system() == 'Linux'   : nl = '\n'

			self.checklist = { \
								'kappa_onx': self.kappa_onx, \
								'kappa_ony': self.kappa_ony, \
								'kappa_onz': self.kappa_onz, \

								'kappa_offx': self.kappa_offx, \
								'kappa_offy': self.kappa_offy, \
								'kappa_offz': self.kappa_offz, \

								'Esigma_onx': self.Esigma_onx, \
								'Esigma_ony': self.Esigma_ony, \
								'Esigma_onz': self.Esigma_onz, \

								'Hsigma_onx': self.Hsigma_onx, \
								'Hsigma_ony': self.Hsigma_ony, \
								'Hsigma_onz': self.Hsigma_onz, \

								'Esigma_offx': self.Esigma_offx, \
								'Esigma_offy': self.Esigma_offy, \
								'Esigma_offz': self.Esigma_offz, \

								'Hsigma_offx': self.Hsigma_offx, \
								'Hsigma_offy': self.Hsigma_offy, \
								'Hsigma_offz': self.Hsigma_offz, \

								'space_eps_on' : self.space_eps_on , \
								'space_eps_off': self.space_eps_off, \

								'space_mu_on' : self.space_mu_on , \
								'space_mu_off': self.space_mu_off, \

								'CDx1': self.CDx1, \
								'CDx2': self.CDx2, \
								'CEx1': self.CEx1, \
								'CEx2': self.CEx2, \
								'CEx3': self.CEx3, \

								'CDy1': self.CDy1, \
								'CDy2': self.CDy2, \
								'CEy1': self.CEy1, \
								'CEy2': self.CEy2, \
								'CEy3': self.CEy3, \

								'CDz1': self.CDz1, \
								'CDz2': self.CDz2, \
								'CEz1': self.CEz1, \
								'CEz2': self.CEz2, \
								'CEz3': self.CEz3, \

								'CBx1': self.CBx1, \
								'CBx2': self.CBx2, \
								'CHx1': self.CHx1, \
								'CHx2': self.CHx2, \
								'CHx3': self.CHx3, \

								'CBy1': self.CBy1, \
								'CBy2': self.CBy2, \
								'CHy1': self.CHy1, \
								'CHy2': self.CHy2, \
								'CHy3': self.CHy3, \

								'CBz1': self.CBz1, \
								'CBz2': self.CBz2, \
								'CHz1': self.CHz1, \
								'CHz2': self.CHz2, \
								'CHz3': self.CHz3, \

							}
			try : import h5py
			except ImportError as e: print("Please install h5py and hdfviewer.")

			f = h5py.File(path+'pml_array_set.h5', 'w')
			for key, value in self.checklist.items(): f.create_dataset(key, data=value)
			f.close()

		self.comm.Barrier()
		return
