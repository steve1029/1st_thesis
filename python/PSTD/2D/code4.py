import numpy as np
from scipy.constants import c, mu_0,epsilon_0
import datetime, time, os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
	Author : Dong Gun Lee
	Purpose : testing 2D PSTD TEz algorithm with UPML.
"""

# print os.listdir('.')
def makefolder(directory):
	
	try :
		os.mkdir(directory)
	except Exception as err :
		# print "%s Directory already exists" %folder
		return "%s Directory already exists" %directory

	return '%s is created' %directory
makefolder('./2D_Plot')
makefolder('./graph')
makefolder('./2D+3D Plot')
makefolder('./3D_Plot')
################################################################################
################################ Space Setting #################################
################################################################################

IE = 108
JE = 108

npml = 10

IEp = IE + 2 * npml
JEp = JE + 2 * npml

ic = int(IEp/2)
jc = int(JEp/2)

nm = 1e-9

wavelength = np.arange(400,800,.5) * nm
wl0 = (wavelength[0] + wavelength[-1])/2.
freq = (c/wavelength)
freqc = (freq[0] + freq[-1])/2
print('center wavelength : %g' %wl0)
print('center frequency : %g' %freqc)

S  = 1./2/np.sqrt(2)
# S = 1./4
# S = 1./8
dx = 5*nm
dy = 5*nm

print('dx = %g meter' %dx)

dt = S*min(dx,dy)/c
print('dt : %g ' %dt)

ft  = np.fft.fftn
ift = np.fft.ifftn
nax = np.newaxis

kx = np.fft.fftfreq(IEp,dx) * 2 * np.pi
ky = np.fft.fftfreq(JEp,dy) * 2 * np.pi

###########################################################################
################################ PML setting ##############################
###########################################################################

kappa_x = np.ones((IEp,JEp),dtype=float)
kappa_y = np.ones((IEp,JEp),dtype=float)
kappa_z = np.ones((IEp,JEp),dtype=float)

sigma_x = np.zeros((IEp,JEp),dtype=float)
sigma_y = np.zeros((IEp,JEp),dtype=float)
sigma_z = np.zeros((IEp,JEp),dtype=float)

kappa_mx = np.ones((IEp,JEp),dtype=float)
kappa_my = np.ones((IEp,JEp),dtype=float)
kappa_mz = np.ones((IEp,JEp),dtype=float)

sigma_mx = np.zeros((IEp,JEp),dtype=float)
sigma_my = np.zeros((IEp,JEp),dtype=float)
sigma_mz = np.zeros((IEp,JEp),dtype=float)

### Grading of the PML Loss parameters ###
rc0 = 1.e-16 	# reflection coefficient.
gradingOrder = 3.
impedence = np.sqrt(mu_0/epsilon_0)
boundarywidth_x = npml * dx
boundarywidth_y = npml * dy
sigmax_x = -(gradingOrder+1)*np.log(rc0)/(2*impedence*boundarywidth_x)
sigmax_y = -(gradingOrder+1)*np.log(rc0)/(2*impedence*boundarywidth_y)
kappamax_x = 5.
kappamax_y = 5.

sigmax_mx = sigmax_x * impedence**2
sigmax_my = sigmax_y * impedence**2
kappamax_mx = kappamax_x
kappamax_my = kappamax_y

###########################################################################
############################ Material setting #############################
###########################################################################

epsr = np.ones((IEp,JEp), dtype=float)
mur  = np.ones((IEp,JEp), dtype=float)

radi = 30
xc, yc = 190, jc

condx_circle = 1e-7
condy_circle = 1e-7

eps_circle = 7.6
mu_circle  = 1.
def circle_2D (center,radius,conductivity,epsr,mu_circle):

	xcenter = center[0]
	ycenter = center[1]

	condx_circle = conductivity[0]
	condy_circle = conductivity[1]	

	for j in range(JEp):
		for i in range(IEp):
			distance = np.sqrt((i-xcenter)**2 + (j-ycenter)**2)
				
			if distance <= radius:
				sigma_x[i][j] = condx_circle
				sigma_y[i][j] = condy_circle
				kappa_x[i][j] = eps_circle
				kappa_y[i][j] = eps_circle
				kappa_z[i][j] = eps_circle
				kappa_mx[i][j]  = mu_circle
				kappa_my[i][j]  = mu_circle
				kappa_mz[i][j]  = mu_circle
def rectangle_2D(lowerleft, width, height, eps_r, mu_r, conductivity):
	"""Apply rectangular shape of material.

	Parameters
	--------------
	lowerleft : int tuple
		coordinates of lowerleft of rectangular.

	width : int
		width of rectangle. Actual length is width * dx

	height : int
		height of rectangle. Actual length is height * dy

	eps_r : float, tuple
		relative electric constant(permitivity) of rectangle.

	mu_r : float, tuple
		relative magnetic constant(permeability) of rectangle.

	conductivity : float
		electric conductivity of rectangle.

	Returns
	-------------
	None

	"""
	if type(eps_r) == int:
		eps_r = float(eps_r)
	elif type(eps_r) == float:
		epsr_x = eps_r
		epsr_y = eps_r
	elif type(eps_r) == tuple:
		eps_rx = eps_r[0]
		eps_ry = eps_r[1]

	if type(mu_r) == float:
		mu_rx = mu_r
		mu_ry = mu_r
	elif type(mu_r) == int:
		mu_rx = float(mu_r)
		mu_ry = float(mu_r)
	elif type(mu_r) == tuple:
		mu_rx = mu_r[0]
		mu_ry = mu_r[1]

	if type(conductivity) == int:
		conductivity_x = float(conductivity)
		conductivity_y = float(conductivity)
	elif type(conductivity) == float:
		conductivity_x = conductivity
		conductivity_y = conductivity
	elif type(conductivity) == tuple:
		conductivity_x = conductivity[0] / eps_rx
		conductivity_y = conductivity[1] / eps_ry 

	xcoor = lowerleft[0]
	ycoor = lowerleft[1]

	for i in range(width):
		for j in range(height):
			epsr[xcoor+i][ycoor+j] = eps_r
			mur [xcoor+i][ycoor+j] = mu_r
			sigma_x[xcoor+i][ycoor+j] = conductivity_x
			sigma_y[xcoor+i][ycoor+j] = conductivity_y

# circle_2D((xc,yc),radi,(condx_circle,condy_circle),eps_circle,mu_circle)
rectangle_2D((170,0),40,JEp,4.,1.,0.)
rectangle_2D((210,0),80,JEp,9.,1.,0.)
eps = epsr * epsilon_0
mu  = mur * mu_0

##################################################################################
################################### Appying PML ##################################
##################################################################################
def apply_PML_2D(**kwargs):
	"""Apply PML Boundary condition along x and y direction.

	Before using the function, following parameters must be defined.

	npml, impedence, GradingOrder,
	sigma_x, sigma_mx
	kappa_x, kappa_mx
	sigma_y, sigma_my
	kappa_y, kappa_my

	KWARGS
	----------

	x : string
		Apply PML along x direction
		
	y : string
		Apply PML along y direction

	Example
	----------
	apply_PML(x = '+', y = '-')

	PML is now applied at x = IE ~ IEp, y = 0 ~ npml

	"""
	for key, value in kwargs.items():
		if key == 'x':
			x = value

		elif key == 'y':
			y = value

		elif key == 'z':
			z = value

	for j in range(JEp):
		
		if x == '-' :
			# if j >= npml and j <= JE+npml-1 :
				# kappa_x[IEp,j]  = 1.e12	# apply PEC on the opposite PML region.
				# kappa_mx[IE+npml,j] = 1.e9	# apply PMC on the opposite PML region.
				
			for i in range(npml):
				sigma_x[i,j]  = sigmax_x * (float(npml-i)/npml)**gradingOrder
				kappa_x[i,j]  = 1 + ((kappamax_x-1)*((float(npml-i)/npml)**gradingOrder))
				sigma_mx[i,j] = sigma_x[i][j] * impedence**2
				kappa_mx[i,j] = kappa_x[i,j]
			
		elif x == '+' :

			# if j >= npml and j <= JE+npml-1 :
				# kappa_x[0,j] = 1.e12	# apply PEC on the opposite PML region.
				# kappa_mx[npml-1,j]= 1.e9	# apply PMC on the opposite PML region.

			for i in range(npml):
				sigma_x[-i-1,j] = sigmax_x * (float(npml-i)/npml)**gradingOrder
				kappa_x[-i-1,j] = 1 + ((kappamax_x-1)*((float(npml-i)/npml)**gradingOrder))
				sigma_mx[-i-1,j]= sigma_x[-i-1,j] * impedence**2
				kappa_mx[-i-1,j]= kappa_x[-i-1,j]

		elif x == '+-':

			for i in range(npml):
				sigma_x[i,j]  = sigmax_x * (float(npml-i)/npml)**gradingOrder
				kappa_x[i,j]  = 1 + ((kappamax_x-1)*((float(npml-i)/npml)**gradingOrder))
				sigma_mx[i,j] = sigma_x[i,j] * impedence**2
				kappa_mx[i,j] = kappa_x[i,j]

				sigma_x[-i-1,j]  = sigma_x[i,j]
				kappa_x[-i-1,j]  = kappa_x[i,j]
				sigma_mx[-i-1,j] = sigma_mx[i,j]
				kappa_mx[-i-1,j] = kappa_mx[i,j]

	for i in range(IEp):

		if y == '-' :
			# if i >= npml and i <= IE+npml-1 :
				# kappa_y[i,JEp] = 1.e12	# apply PEC on the opposite PML region.
				# kappa_my[i,JE+npml]= 1.e9	# apply PMC on the opposite PML region.

			for j in range(npml):
				sigma_y[i,j]    = sigmax_y * (float(npml-j)/npml)**gradingOrder
				kappa_y[i,j]	 = 1 + ((kappamax_y-1)*((float(npml-j)/npml)**gradingOrder))
				sigma_my[i,j]	 = sigma_y[i,j] * impedence**2
				kappa_my[i,j]	 = kappa_y[i,j]

		elif y == '+' :

			# if i >= npml and i <= IE+npml-1 :
				# kappa_y[i,0] = 1.e12	# apply PEC on the opposite PML region.
				# kappa_my[i,npml-1]= 1.e9	# apply PMC on the opposite PML region.
	
			for j in range(npml):
				sigma_y[i,-j-1] = sigmax_y * (float(npml-j)/npml)**gradingOrder
				kappa_y[i,-j-1] = 1 + ((kappamax_y-1)*((float(npml-j)/npml)**gradingOrder))
				sigma_my[i,-j-1]= sigma_y[i,-j-1] * impedence**2
				kappa_my[i,-j-1]= kappa_y[i,-j-1]

		elif y == '+-':

			for j in range(npml):
				sigma_y[i,j]  = sigmax_y * (float(npml-j)/npml)**gradingOrder
				kappa_y[i,j]  = 1 + ((kappamax_y-1)*((float(npml-j)/npml)**gradingOrder))
				sigma_my[i,j] = sigma_y[i,j] * impedence**2
				kappa_my[i,j] = kappa_y[i,j]

				sigma_y[i,-j-1] = sigmax_y * (float(npml-j)/npml)**gradingOrder
				kappa_y[i,-j-1] = 1 + ((kappamax_y-1)*((float(npml-j)/npml)**gradingOrder))
				sigma_my[i,-j-1]= sigma_y[i,-j-1] * impedence**2
				kappa_my[i,-j-1]= kappa_y[i,-j-1]

def apply_PEC_2D(**kwargs):
	"""Apply PEC Boundary condition along x and y direction.

	Before using the function, following parameters must be defined.

	npml, impedence, GradingOrder,
	sigma_x, sigma_mx
	kappa_x, kappa_mx
	sigma_y, sigma_my
	kappa_y, kappa_my

	KWARGS
	----------

	x : string
		Apply PEC along x direction
		
	y : string
		Apply PEC along y direction

	Example
	----------
	apply_PML(x = '+', y = '-')

	PML is now applied at x = IE ~ IEp, y = 0 ~ npml

	"""
	for key, value in kwargs.items():
		if key == 'x':
			x = value

		elif key == 'y':
			y = value

		elif key == 'z':
			z = value

	for j in range(JEp):
		if x == '-':
			kappa_x[0,j]  = 1.e12	# apply PEC on the opposite PML region.

		elif x == '+':
			kappa_x[-1,j] = 1.e12	

		elif x == '+-':
			kappa_x[-1,j]  = 1.e12
			kappa_x[0,j] = 1.e12	

	for i in range(IEp):
		if y == '-':
			kappa_y[i,0] = 1.e12

		elif y == '+':
			kappa_y[i,-1] = 1.e12

		elif y == '+-':
			kappa_y[i,0] = 1.e12
			kappa_y[i,-1] = 1.e12
PML = True
# PML = False
PEC = True
# PEC = False

if PML == True:
	apply_PML_2D(x='+-', y='')
if PEC == True:
	apply_PEC_2D(x='', y='')

# setting kappa and sigma for PML

######################################################################################
############################## Preparing Coefficients ################################
######################################################################################
px = (2*epsilon_0*kappa_x) + (sigma_x * dt)
py = (2*epsilon_0*kappa_y) + (sigma_y * dt)
pz = (2*epsilon_0*kappa_z) + (sigma_z * dt)

mx = (2*epsilon_0*kappa_x) - (sigma_x * dt)
my = (2*epsilon_0*kappa_y) - (sigma_y * dt)
mz = (2*epsilon_0*kappa_z) - (sigma_z * dt)

mpx = (2*mu_0*kappa_mx) + (sigma_mx * dt)
mpy = (2*mu_0*kappa_my) + (sigma_my * dt)
mpz = (2*mu_0*kappa_mz) + (sigma_mz * dt)

mmx = (2*mu_0*kappa_mx) - (sigma_mx * dt)
mmy = (2*mu_0*kappa_my) - (sigma_my * dt)
mmz = (2*mu_0*kappa_mz) - (sigma_mz * dt)

######################################################################################
################################ Cheking physical parameters #########################
######################################################################################
dataset = ['kappa_x','kappa_y','sigma_x','sigma_y',\
			'px','py','pz','mx','my','mz',\
			'kappa_mx','kappa_my','sigma_mx','sigma_my',\
			'mpx','mpy','mpz','mmx','mmy','mmz']

def check_coefficient(dataset):

	for data in dataset:
		if PML == True:
			f = open('./%s.txt' %data,'w')
		elif PML == False:
			f = open('./%s no PML.txt' %data,'w')

		f.write('%s : \n' %data)				
		for i in range(IEp):						
			for j in range(JEp):				
				if data == 'kappa_x':				
					f.write('%.3g\t\t' %(kappa_x[i][j]))
				elif data == 'kappa_y':				
					f.write('%.3g\t\t' %(kappa_y[i][j]))
				elif data == 'sigma_x':				
					f.write('%.3g\t\t' %(sigma_x[i][j]))
				elif data == 'sigma_y':				
					f.write('%.3g\t\t' %(sigma_y[i][j]))
				elif data == 'px':
					f.write('%.3g\t\t' %(px[i][j]))
				elif data == 'py':
					f.write('%.3g\t\t' %(py[i][j]))
				elif data == 'pz':
					f.write('%.3g\t\t' %(pz[i][j]))
				elif data == 'mx':
					f.write('%.3g\t\t' %(mx[i][j]))
				elif data == 'my':
					f.write('%.3g\t\t' %(my[i][j]))
				elif data == 'mz':
					f.write('%.3g\t\t' %(mz[i][j]))

				elif data == 'kappa_mx':
					f.write('%.3g\t\t' %(kappa_mx[i][j]))	
				elif data == 'kappa_my':				
					f.write('%.3g\t\t' %(kappa_my[i][j]))
				elif data == 'kappa_mz':				
					f.write('%.3g\t\t' %(kappa_mz[i][j]))
				elif data == 'sigma_mx':				
					f.write('%.3g\t\t' %(sigma_mx[i][j]))
				elif data == 'sigma_my':				
					f.write('%.3g\t\t' %(sigma_my[i][j]))
				elif data == 'sigma_mz':				
					f.write('%.3g\t\t' %(sigma_mz[i][j]))
				elif data == 'mpx':
					f.write('%.3g\t\t' %(mpx[i][j]))
				elif data == 'mpy':
					f.write('%.3g\t\t' %(mpy[i][j]))
				elif data == 'mpz':
					f.write('%.3g\t\t' %(mpz[i][j]))
				elif data == 'mmx':
					f.write('%.3g\t\t' %(mmx[i][j]))
				elif data == 'mmy':
					f.write('%.3g\t\t' %(mmy[i][j]))
				elif data == 'mmz':
					f.write('%.3g\t\t' %(mmz[i][j]))
			f.write('\n')							
	f.close()		

check_coefficient(dataset)
##############################################
################ Field Arrays ################
##############################################

Ez = np.zeros((IEp,JEp),dtype=complex)
Dz = np.zeros((IEp,JEp),dtype=complex)

Hx = np.zeros((IEp,JEp),dtype=complex)
Hy = np.zeros((IEp,JEp),dtype=complex) 

Bx = np.zeros((IEp,JEp),dtype=complex)
By = np.zeros((IEp,JEp),dtype=complex)

##############################################
################## Figure ####################
##############################################
def plot2D3Din2D(field,**kwargs):
	"""
	"""
	colordeep = 1.
	stride = 1
	zlim = 1.
	for key, item in kwargs.items():
		if key == 'colordeep':
			colordeep = item
		elif key == 'stride':
			stride = item
		elif key == 'zlim':
			zlim = item

	if field == 'Ez':
		plotfield = Ez
	plotfield = plotfield.T[::-1,:]
	
	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import datetime

	x = np.arange(IEp)
	y = np.arange(JEp)
	Y, X = np.meshgrid(x,y)
	today = datetime.date.today()
	
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122,projection='3d')

	im = ax1.imshow(plotfield[:,:].real, vmax=colordeep, vmin=-colordeep, cmap=plt.cm.bwr)
	ax1.set_title('%s, 2D plot' %str(field))
	ax1.set_xlabel('x')
	ax2.set_ylabel('y')
	divider = make_axes_locatable(ax1)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = fig.colorbar(im,cax=cax)
	ylabels = ax1.get_yticks().tolist()
	ax1.set_yticklabels(ylabels[::-1])

	ax2.plot_wireframe(Y,X,plotfield[X,Y].real,color='b',rstride=stride, cstride=stride)
	ax2.set_title('%s, 3D plot' %str(field))
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_zlim(-zlim,zlim)
	fig.savefig('./2D+3D Plot/%s %s' %(str(today),step))
	fig.clf()

def plot2Din2D(field,colordeep):

	if field == 'Ez':
		plotfield = Ez

	plotfield = plotfield.T[::-1,:] # take transpose and reverse the order of x axis.

	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import datetime
	today = datetime.date.today()

	ax = fig.add_subplot(111)
	im = ax.imshow(plotfield[:,:].real, vmax=colordeep, vmin=-colordeep, cmap=plt.cm.bwr)
	ax.set_title('%s, 2D plot' %str(field))
	ax.set_xlabel('x')
	ax.set_ylabel('y')

	ylabels = ax.get_yticks().tolist()
	ax.set_yticklabels(ylabels[::-1])

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = fig.colorbar(im,cax=cax)
	fig.savefig('./2D_Plot/%s %s' %(str(today), step))
	fig.clf()

fig = plt.figure(figsize=(21,9))
plt.ion()
##############################################
################### Source ###################
##############################################

w0 = 2 * np.pi * freqc
ws = 0.3 * w0
print('ws : %g' %ws)
ts = 1./ws
tc = 1000.*dt

src_xpos = 10 + npml
# src_ypos = jc

trans_xpos = -10 - npml
# trans_ypos = jc

##############################################
################ Main FDTD Loop ##############
##############################################

nsteps = 15001

Ez_src = np.zeros(nsteps, dtype=complex)
Ez_ref = np.zeros(nsteps, dtype=complex)
Ez_trs = np.zeros(nsteps, dtype=complex)

file_src = open('./graph/Ez_src.txt','w')
file_ref = open('./graph/Ez_ref.txt','w')
file_trs = open('./graph/Ez_trs.txt','w')

tstart = time.time()

print( "Simulation Start")

for step in range(nsteps):

	# make source
	pulse = (np.exp((-.5)*(((step*dt-tc)*ws)**2)) * np.exp(-1.j*w0*(step*dt-tc))).real
	# pulse = np.exp((-.5)*(((step*dt-tc)*ws)**2))
	# pulse = np.sin(2*np.pi*freqc*step*dt)

	# Ez[65,65] += pulse
	for j in range(JEp):
		Ez[src_xpos,j] += pulse

	Ez_src[step] = Ez[src_xpos,:].mean()
	Ez_ref[step] = Ez[src_xpos,:].mean() - pulse /S/2
	Ez_trs[step] = Ez[trans_xpos,:].mean()

	file_src.write('%d\t %.5f\t %.5f \n' %(step, Ez_src[step].real, abs(Ez_src[step].real)**2))
	file_ref.write('%d\t %.5f\t %.5f \n' %(step, Ez_ref[step].real, abs(Ez_ref[step].real)**2))
	file_trs.write('%d\t %.5f\t %.5f \n' %(step, Ez_trs[step].real, abs(Ez_trs[step].real)**2))

	# update Hy field
	previous = By
	diffxEz = ift(1j*kx[:,nax]*ft(Ez,axes=(0,1)),axes=(0,1))
	By = (mmz/mpz) * By - (2*mu_0*dt/mpz)*(-diffxEz)
	Hy = (mmx/mpx) * Hy + (mpy * By - mmy * previous) / mpx / mu

	# update Hx field
	previous = Bx
	diffyEz = ift(1j*ky[nax,:]*ft(Ez,axes=(0,1)),axes=(0,1))
	Bx = (mmy/mpy) * Bx - (2*mu_0*dt/mpy)*(diffyEz)
	Hx = (mmz/mpz) * Hx + (mpx * Bx - mmx * previous) / mpz / mu

	# update Dz field
	previous = Dz
	diffxHy = ift(1j*kx[:,nax]*ft(Hy,axes=(0,)),axes=(0,))
	diffyHx = ift(1j*ky[nax,:]*ft(Hx,axes=(1,)),axes=(1,))
	Dz = (mx/px) * Dz + (2*epsilon_0*dt/px)*(diffxHy - diffyHx)
	Ez = (my/py) * Ez + (pz * Dz - mz * previous) / py / eps
	
	if step % 1000 == 0:

		plot2D3Din2D('Ez',colordeep=2.,stride=4, zlim=2.)
		# plot2Din2D('Ez',2.)
		tend = time.time() - tstart
		print('time : %.2f (s), steps : %d' % (tend,step))

print("Simulation Complete")
plt.ioff()

tsteps = np.arange(nsteps, dtype=float)
t = tsteps * dt
# Ez_src = (ws * np.exp((-.5)*(((tsteps-tc)*dt*ws)**2)) * np.exp(-1.j*w0*(tsteps-tc)*dt)).real
Ez_Ori = Ez_src
Ez_src = (np.exp((-.5)*(((tsteps*dt-tc)*ws)**2)) * np.exp(-1.j*w0*(tsteps*dt-tc))).real /S/2

# Ez_src_ft = ft(Ez_src)
# Ez_ref_ft = ft(Ez_ref)
# Ez_trs_ft = ft(Ez_trs)
# freq = np.fft.fftfreq(nsteps,dt)

Ez_Ori_ft = (dt * Ez_Ori[nax,:]*np.exp(1.j*2.*np.pi*freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
Ez_src_ft = (dt * Ez_src[nax,:]*np.exp(1.j*2.*np.pi*freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
Ez_ref_ft = (dt * Ez_ref[nax,:]*np.exp(1.j*2.*np.pi*freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
Ez_trs_ft = (dt * Ez_trs[nax,:]*np.exp(1.j*2.*np.pi*freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)

###########################################################
#################### Save E Field Data ####################
###########################################################

# file_src.write('%s\t %s \n' %('step', 'Ez_src'))
# file_ref.write('%s\t %s \n' %('step', 'Ez_ref'))
# file_trs.write('%s\t %s \n' %('step', 'Ez_trs'))

file_src.close()
file_ref.close()
file_trs.close()

###########################################################
##################### Plot R,T Graph ######################
###########################################################

wl = wavelength
# wl = c/freq
Trans  = (abs(Ez_trs_ft)**2)/(abs(Ez_src_ft)**2)
Reflec = (abs(Ez_ref_ft)**2)/(abs(Ez_src_ft)**2)
Total = Trans + Reflec

RTgraph = plt.figure(figsize=(21,9))
ax1 = RTgraph.add_subplot(121)
ax1.plot(wl/nm, Reflec, label='Ref', color='green')
ax1.plot(wl/nm, Trans, label ='Trs', color='red')
ax1.plot(wl/nm, Total, label ='Total', color='blue')
ax1.set_xlabel('wavelength, nm')
ax1.set_ylabel('Ratio')
ax1.set_ylim(0.,1.1)
ax1.legend(loc='best')
ax1.grid(True)

ax2 = RTgraph.add_subplot(122)
ax2.plot(freq/1.e12, Reflec, label='Ref', color='green')
ax2.plot(freq/1.e12, Trans, label = 'Trs', color='red')
ax2.plot(freq/1.e12, Total, label = 'Total', color='blue')
ax2.set_xlabel('freq, THz')
ax2.set_ylabel('Ratio')
ax2.set_ylim(0.,1.1)
ax2.legend(loc='best')
ax2.grid(True)

RTgraph.savefig('./graph/RT graph.png')

##############################################################
###################### Plot Source Graph #####################
##############################################################

Srcgraph = plt.figure(figsize=(31,9))

ax1 = Srcgraph.add_subplot(131)
ax1.plot(freq/1.e12, abs(Ez_src_ft)**2, label='$E_{src}(\omega)$')
ax1.plot(freq/1.e12, abs(Ez_ref_ft)**2, label='$E_{ref}(\omega)$')
ax1.plot(freq/1.e12, abs(Ez_trs_ft)**2, label='$E_{trs}(\omega)$')
ax1.set_xlabel('freq, THz')
ax1.set_ylabel('ratio')
ax1.legend(loc='best')
ax1.grid(True)

ax2 = Srcgraph.add_subplot(132)
ax2.plot(wl/nm,abs(Ez_src_ft)**2, label='$E_{src}(\lambda)$')
ax2.plot(wl/nm,abs(Ez_ref_ft)**2, label='$E_{ref}(\lambda)$')
ax2.plot(wl/nm,abs(Ez_trs_ft)**2, label='$E_{trs}(\lambda)$')
ax2.set_xlabel('wavelength, nm')
ax2.legend(loc='best')
ax2.grid(True)

ax3 = Srcgraph.add_subplot(133)
divider = make_axes_locatable(ax3)
ax3.plot(tsteps[0:int(4*tc/dt)],(abs(Ez_src)**2)[0:int(4*tc/dt)], label='$|Ez(t)|^2|,Ori$')
ax3.plot(tsteps[0:int(4*tc/dt)],(abs(Ez_Ori)**2)[0:int(4*tc/dt)], label='$|Ez(t)|^2|,Thm$')
ax3.get_xaxis().set_visible(False)
ax3.set_title('Input Source, $t_c = %gdt $' %(tc/dt))
ax3.text(2000,1.5,'$E(t)=e^{-\\frac{1}{2}(\\frac{t-t_c}{ts})^{2}}\cos(\omega_{0}(t-t_c))$', fontsize=18)
ax3.set_ylim(0,)
ax3.legend()
ax3.grid(True)

ax4 = divider.append_axes("bottom",size="100%",pad=0.2, sharex=ax3)
ax4.plot(tsteps[0:int(4*tc/dt)],(abs(Ez_ref)**2)[0:int(4*tc/dt)], label='$|E_r(t)|^2$',color='green')
ax4.plot(tsteps[0:int(4*tc/dt)],(abs(Ez_trs)**2)[0:int(4*tc/dt)], label='$|E_t(t)|^2$',color='red')
ax4.set_ylim(0,)
ax4.set_xlabel('time step, dt=%.2g' %(dt))
ax4.legend()
ax4.grid(True)
Srcgraph.savefig('./graph/Srcgraph.png')

Src_E_t = plt.figure()
ax = Src_E_t.add_subplot(111)
ax.plot(tsteps[0:int(2*tc/dt)],(abs(Ez_src)**2)[0:int(2*tc/dt)], label='$|Ez(t)|^2|$')
ax.set_title('Input Source, $t_c = %gdt $' %(tc/dt))
ax.set_xlabel('time steps')
ax.set_ylabel('$|E(t)|^2$')
ax.set_ylim(0,16)
ax.legend()
ax.grid(True)
Src_E_t.savefig('./graph/Src_E_t.png')
############################################################################
############### PLOT Theoratical graph and Simulation graph ################
############################################################################

from newIRT import IRT

TMM = IRT()
TMM.wavelength(wl)
TMM.incidentangle(angle=0, unit='radian')
TMM.mediumindex(1.,2.,3.,1.)
TMM.mediumtype('nonmagnetic')
TMM.mediumthick(200*nm, 400*nm)
TMM.cal_spol_matrix()
TMMref = TMM.Reflectance()
TMMtrs = TMM.Transmittance()
TMMfreq = TMM.frequency

fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(wl/nm, TMMref, color='red', 		alpha=.5, linewidth=3, label='Theoratical')
ax2.plot(wl/nm, TMMtrs, color='red', 		alpha=.5, linewidth=3, label='Theoratical')
ax3.plot(freq/1.e12, TMMref, color='red',	alpha=.5, linewidth=3, label='Theoratical')
ax4.plot(freq/1.e12, TMMtrs, color='red',	alpha=.5, linewidth=3, label='Theoratical')

ax1.plot(wl/nm, Reflec, 	label='Simulation')
ax2.plot(wl/nm, Trans, 		label='Simulation')
ax3.plot(freq/1.e12, Reflec,label='Simulation')
ax4.plot(freq/1.e12, Trans, label='Simulation')

ax1.set_xlabel('wavelength')
ax2.set_xlabel('wavelength')
ax3.set_xlabel('frequency')
ax4.set_xlabel('frequency')

ax1.set_ylabel('ratio')
ax2.set_ylabel('ratio')
ax3.set_ylabel('ratio')
ax4.set_ylabel('ratio')

ax1.set_title('Reflectance')
ax2.set_title('Transmittance')
ax3.set_title('Reflectance')
ax4.set_title('Transmittance')

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax4.legend(loc='best')

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)

ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
ax3.set_ylim(0,1)
ax4.set_ylim(0,1)

fig.savefig('./graph/Theory vs Simul.png')
print("Simulation finished : " , datetime.datetime.now())
