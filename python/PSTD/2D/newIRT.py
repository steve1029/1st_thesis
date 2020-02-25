import numpy as np
from numpy.linalg import inv 
import matplotlib.pyplot as plt
import atexit,sys
from scipy.constants import c, mu_0,epsilon_0
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
import datetime, time, os
from functools import reduce

class IRT():

	"""Calculate Reflectance and Transmittance for p pol and s pol.

	Waveguide is lying along z direction.
	I and R denote amplitude of electric field of incident light and reflected light.
	T denotes amplitude of electric field of transmitted light.


					|				|				|			|				|
		I ------->	| A1 --------->	| A2 --------->	| 			| An --------->	| T --------->
					|				|				| ...	...	|				|
		R <-------	| B1 <---------	| B2 <---------	|			| Bn <---------	|
					|				|				|			|				|

	--------------------------- z direction ---------------------->

	"""
	########################################################
	###################### Setting up ######################
	########################################################
	
	def __init__(self):
		self.incangle = 0.
		self.incEamp = 1.
		self.startmu = 1.
		self.endmu = 1.

	def wavelength(self, wavelength):
		"""Define the range of spectrum.

		Input the range of wavelength to calculate.

		PARAMETERS
		--------------
		wavelength : ndarray
			the array or list of wavelength. ex) 400 nm ~ 800 nm

		RETURN
		--------------
		None

		"""
		self.wavelength = wavelength
		self.wavevector = 2 * np.pi / wavelength
		self.frequency = c/self.wavelength

		# print self.wavevector

		return None

	def incidentangle(self, angle,**kwargs):
		
		"""Set incident angle.

		PARAMETERS
		----------
		angle 	: float
			incident angle


		KEYWORD ARGUMENTS
		-----------------
		unit	: string
			specify the unit of incident angle you type.
			Default unit is radian.

		RETURN
		-------
		None
		"""
		# print kwargs.keys()
		# print kwargs.values()

		for key in kwargs:
			# print key
			if key == 'unit':
				if kwargs[key] == 'radian':
					# print 'radian!'
					self.incangle = angle

				elif kwargs[key] == 'degree':
					self.incangle = angle * np.pi/180
					# print 'degree!'
				else:
					raise NotImplementedError

		return self.incangle

	def mediumindex(self, *arg):
		"""Input index of mediums

		Suppose that system has 5 mediums. We call first one as is start medium and
		last one as end medium. Then, middle medium is consist of 3 mediums.
		You must input 5 values of index. 

		Note : As you can see in documentation of
		mediumthick medium, we need index of all medium but 
		in case of thcikness, thickness of middle medium is only needed.
		"""
		self.mediumindex = np.array(arg)
		return None

	def mediumtype(self,mtype,*arg):
		self.mtype = mtype

		"""Setting up material property.

		If all mediums in system are nonmagnetic, calculator needs only index.
		It automatically set relative magnetic constant(relative permeability) as 1.

		However, if medium has at least 1 magnetic medium, one must input
		the magnetic constants of all mediums, using this method.

		PARAMETERS
		------------

		mtype : string
			'nonmagnetic' or 'magnetic'

		arg : ndarray, list, tuple
			ralative permeability(magnetic constant) of each medium.

		RETURNS
		----------
		None
		"""

		try:
			if mtype == 'nonmagnetic' or mtype == None:
				self.mediummur = np.ones(len(self.mediumindex))
				print('material type : nonmagnetic')
				# return self.mediummur
				return None
			elif mtype == 'magnetic':
				# self.mediummur = np.ones(len(self.mediumindex))
				# set medium mu_r
				self.mediummur = np.array(arg)
				print('material type : magnetic')
				# return self.mediummur
				return None
		except Exception as error :
			print( 'Error detected : ',error)
			print( 'mediumindex method must be called before mediumtype method.')
			# raise NotImplementedError
			sys.exit()

	def mediumthick(self,*arg):

		"""Thickness of mediums.

		Thickness of first medium and last medium is unnecessary.
		Just put the thickness of inside medium. 
		For example, if you had	put 5 index by self.medium method, self.mediumthick method only need 3 values.

		PARAMETERS
		------------
		None

		ARGUMENTS
		------------
		thiscklist	: ndarray
			array of thickness of inside medium.
		"""
		self.mediumthick = np.array(arg)
		return None

	def inctEamp(self,I):
		"""Set up the amplitude of incident E field

		Actually it isn't necessary, but if you want to get
		amplitude of reflected and transmitted E field according to
		the specific value of amplitude of incident wave, use this method
		"""
		self.incEamp = I

	#########################################################
	################### Calculation Method ##################
	#########################################################
	def cal_normal_matrix(self):

		self.matrixN = Ms

		# return self.matrixN

	def cal_spol_matrix(self):
		"""Obtain the system matrix between [i, r] and [t,0] for s polarized incident light.
		
		System matrix is defined by equation such that

		[i ,r] = np.dot(M,[t,0])

		i is amplitude of input E field
		r is amplitude of reflected E field
		t is amplitude of transmitted E field

		"""

		anglelist = [self.incangle]
		index = self.mediumindex
		mur = np.ones(len(index))
		d = self.mediumthick
		k0 = self.wavevector

		#####################################################################
		#### Obtain incident angles at each interfaces using Snell's law ####
		#####################################################################
		
		for i in range(len(index)-1):
			theta = np.arcsin(index[i] * np.sin(anglelist[i]) / index[i+1])
			# print anglelist[i]*180/np.pi
			anglelist.append(theta)

		print('angle list : ', (np.array(anglelist) * 180/np.pi))
		
		#####################################################################
		################### Calculate relative permitivity ##################
		###################     and relative impedence	   ##################
		#####################################################################

		reps = (index**2)/mur 		# relative epsilon
		# print 'Relative epsilon : ',reps
		rimp = np.sqrt(mur/reps)	# relative impedence

		#####################################################################
		########################## Get matrix M #############################
		#####################################################################

		Mslist = []

		for n, wv in enumerate(k0):
			
			Dlist = []
			Plist = []
			Ms = np.identity(2,dtype=complex)

			for i in range(len(index)):
				z = np.cos(anglelist[i])/rimp[i]
				D = np.array([[1,1],[z,-z]])
				Dlist.append(D)
				# if i==1 : print z
			for i in range(len(index)-2):
				p = np.exp(1j*index[i+1]*wv*d[i]*np.cos(anglelist[i]))
				P = np.array([[1/p,0],[0,p]])
				Plist.append(P)

			for i in range(len(index)-2):
				Ms = reduce(np.dot,[Ms,Dlist[i+1],Plist[i],inv(Dlist[i+1])])
				# if i==1 : print Ms
				# print i
			Ms = reduce(np.dot,[inv(Dlist[0]),Ms,Dlist[-1]])

			Mslist.append(Ms)

			# print 'n : ', n

		self.matrixs = Mslist
		self.polarization = 'S pol'

		return self.matrixs

	def cal_ppol_matrix(self):

		"""Obtain the system matrix between [i, r] and [t,0] for p polarized incident light.
		
		System matrix is defined by equation such that

		[i ,r] = np.dot(M,[t,0])

		i is amplitude of input E field
		r is amplitude of reflected E field
		t is amplitude of transmitted E field

		"""

		anglelist =[self.incangle]
		index = self.mediumindex
		mur = np.ones(len(index))
		d = self.mediumthick
		k0 = self.wavevector

		#####################################################################
		#### Obtain incident angles at each interfaces using Snell's law ####
		#####################################################################
		
		for i in range(len(index)-1):
			theta = np.arcsin(index[i] * np.sin(anglelist[i]) / index[i+1])
			anglelist.append(theta)

		print('angle list : ', np.array(anglelist) * 180/np.pi)
		
		#####################################################################
		################### Calculate relative permitivity ##################
		###################     and relative impedence	   ##################
		#####################################################################

		reps = (index**2)/mur 		# relative epsilon
		rimp = np.sqrt(mur/reps)	# relative impedence

		#####################################################################
		########################## Get matrix M #############################
		#####################################################################

		Mplist = []

		for n, wv in enumerate(k0):
			
			Dlist = []
			Plist = []
			Mp = np.identity(2,dtype=complex)

			for i in range(len(index)):
				# z = (1/rimp[i])*np.cos(anglelist[i])
				# D = np.array([[1,1],[z,-z]])
				z = 1/rimp[i]
				D = np.array([[np.cos(anglelist[i]), np.cos(anglelist[i])],[z,-z]])
				Dlist.append(D)
				# if i==1 : print z
			for i in range(len(index)-2):
				p = np.exp(1j*index[i+1]*wv*d[i]*np.cos(anglelist[i]))
				P = np.array([[1/p,0],[0,p]])
				Plist.append(P)

			for i in range(len(index)-2):
				Mp = reduce(np.dot,[Mp,Dlist[i+1],Plist[i],inv(Dlist[i+1])])
				# if i==1 : print Mp
			Mp = reduce(np.dot,[inv(Dlist[0]),Mp,Dlist[-1]])

			Mplist.append(Mp)
		
		self.matrixp = Mplist
		self.polarization = 'P pol'

		return self.matrixp

	def Reflectance(self):

		if self.polarization == 'S pol':
			m = self.matrixs
			print('Ref for :',self.polarization)
		elif self.polarization == 'P pol':
			m = self.matrixp
			print('Ref for :',self.polarization)

		Rlist = []

		for wl in range(len(self.wavelength)):
			R = abs(m[wl][1,0]/m[wl][0,0])**2
			Rlist.append(R)

		self.Reflect = np.array(Rlist)
		# print self.Reflect
		return self.Reflect

	def Transmittance(self):
		
		if self.polarization == 'S pol':
			m = self.matrixs
			print('Trs for :', self.polarization)
		elif self.polarization == 'P pol':
			m = self.matrixp
			print('Trs for :', self.polarization)

		Tlist = []

		for wl in range(len(self.wavelength)):
			T = abs(1/m[wl][0,0])**2

			Tlist.append(T)

		self.Transmit = np.array(Tlist)
		
		return self.Transmit

	def graph(self,xaxis, **kwargs):
		"""Plot Reflectance, Transmittance graph.

		Parameters
		-------------
		xaxis : string
			Choose xaxis to plot. Ex) 'wavelength' or 'frequency'

		figuresize : tuple
			Define the size of figure. Default size is (10,8)

		Return
		-------------
		None
		"""
		figuresize = (10,8)
		nm = 1.e-9

		if xaxis == 'wavelength':
			xx = self.wavelength / nm
		elif xaxis == 'frequency':
			xx = c/self.wavelength

		for key, item in kwargs.items():
			if key == 'figuresize' or key == 'figsize':
				figuresize = item

		Trans  = self.Transmit
		Reflec = self.Reflect
		Total = Trans + Reflec
		
		fig = plt.figure(figsize=figuresize)
		ax = fig.add_subplot(111)

		ax.plot(xx, Reflec,color='Green',label='Reflec')
		ax.plot(xx, Trans,color='Red',label='Trans')
		ax.plot(xx, Total,color='Blue',label='total')
		ax.set_title('%s, angle : %.3f' 'rad' % (self.polarization, self.incangle))
		ax.legend(loc='best')
		ax.set_ylim(0.,1.1)
		ax.set_xlabel('%s' %xaxis)
		plt.savefig('./%s %s.png' %(self.polarization, xaxis))
		plt.clf()

		return None


if __name__ == '__main__':

	import numpy as np
	from numpy.linalg import inv 
	import matplotlib.pyplot as plt
	import atexit,sys
	from scipy.constants import c, mu_0,epsilon_0

	example = IRT()

	nm = 1.e-9
	micron = 1.e-6
	# freq = np.arange(400,800) * 1e9

	wl = np.arange(100,1000,0.5) * nm

	example.wavelength(wl)

	print('default incident angle : ', example.incangle)

	brewsterangle = np.arctan(2)
	example.incidentangle(angle= 0, unit='radian')
	print('modified incident angle : ', example.incangle)
	# print '%.2e' %example.wavelength

	print(example.mediumindex(1.,2.,3.,1.))
	# print example.mediumtype('magnetic',[1,2,3,2,1])
	print(example.mediumtype('nonmagnetic'))
	print(example.mediumthick(200*nm,400*nm))

	size = (18,7)

	matrixs = example.cal_spol_matrix()
	ms = example.matrixs
	Reflecs = example.Reflectance() 
	Transms = example.Transmittance()
	example.graph('wavelength',figsize=size)
	example.graph('frequency',figuresize=size)
	matrixp = example.cal_ppol_matrix()
	mp = example.matrixp
	Reflecp = example.Reflectance() 
	Transmp = example.Transmittance()
	example.graph('wavelength',figuresize=size)
	example.graph('frequency',figsize=size)
	# print np.array(mp)
	# print Reflecs - Reflecp
	# print Transms - Transmp
	# print example.__doc__
	# print example.cal_ppol_matrix.__doc__