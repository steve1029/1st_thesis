import numpy as np
from scipy.constants import epsilon_0

class Model(object):
	pass


class Plain(Model):
	pass


class Debye(Model):
	pass


class Drude(Model):
	pass


class Lorentz2P(Model):
	pass


class Lorentz3P(Model):
	def __init__(self, Struct, econ, eps_inf, amp_m, omega0_m, delta_m):
		"""Put the parameters for a Lorentz-modeled materials.

		Parameters
		-------------------
		Struct		: Structure object
			Structure object which is defined in structure.py
		econ		: float
			electric conductivity of the material
		eps_inf		: float
			relative epsilon of a material at infinite frequency.
		eps_dc		: array_like
			epsilon of a 'M lorent pole' material when zero frequency wave is incident.
		eps_inf		: array_like
			epsilon of a 'M lorent pole' material when infinite frequency wave is incident.
		omega0_m	: array_like
			omega0_m of a 'M lorent pole' material
		delta		: array_like
			damping factor of 'M lorent pole' materials

		Returns
		--------------------
		message		: string
		"""
		# The number of Lorentz pole
		M = 3

		assert len(amp_m)	 == M
		assert len(omega0_m) == M
		assert len(delta_m)  == M

		self.model	  = 'Lorentz'
		self.eps_inf  = eps_inf
		self.amp_m	  = amp_m
		self.omega0_m = omega0_m
		self.delta_m  = delta_m
		self.econ	  = econ

		self.Space = Struct.Space

		# Auxilary fields
		if Struct.global_loc != None:

			# Pole 1.
			self.Jx1 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.Jy1 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.Jz1 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			
			# Pole 2.
			self.Jx2 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.Jy2 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.Jz2 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			
			# Pole 3.
			self.Jx3 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.Jy3 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.Jz3 = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			
			# E field for (n-1) time step.
			self.En1x = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.En1y = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
			self.En1z = np.zeros(Struct.local_size, dtype=Struct.Space.dtype)
		
			self.A1 = np.zeros(M)
			self.A2 = np.zeros(M)
			self.A3 = np.zeros(M)

			dt = self.Space.dt

			for pole in range(M):
				self.A1[pole] = (2. - omega0_m[pole]**2 * dt**2) / (1. + delta_m[pole]*dt)
				self.A2[pole] = (delta_m[pole]*dt - 1.) / (delta_m[pole]*dt + 1.)
				self.A3[pole] = epsilon_0 * amp_m[pole] * dt**2

			self.C1 = (self.A3.sum() / 2.) / (2. * epsilon_0 * self.eps_inf + self.econ*dt + self.A3.sum()/2.)
			self.C2 = (2. * epsilon_0 * self.eps_inf - self.econ*dt) / (2. * epsilon_0 * self.eps_inf + self.econ*dt + self.A3.sum()/2.)
			self.C3 = (2. * dt)/ (2. * epsilon_0*self.eps_inf + self.econ*dt + self.A3.sum()/2.)

			print("rank {:2d} >>> A1:{}, A2:{}, A3:{}, C1:{}, C2:{}, C3:{}" \
					.format(Struct.Space.MPIrank, self.A1, self.A2, self.A3, self.C1, self.C2, self.C3))


class Critical_Point(Model):
	pass
