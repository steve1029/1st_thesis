import numpy as np

#########################################################################################
################################# PML PARAMETER SETTING  ################################
#########################################################################################

kappa_x = np.ones(IEp, dtype=np.float32)
kappa_y = np.ones(JEp, dtype=np.float32)
kappa_z = np.ones(KEp, dtype=np.float32)

sigma_x = np.ones(IEp, dtype=np.float32)
sigma_y = np.ones(JEp, dtype=np.float32)
sigma_z = np.ones(KEp, dtype=np.float32)

kappa_mx = np.ones(IEp, dtype=np.float32)
kappa_my = np.ones(JEp, dtype=np.float32)
kappa_mz = np.ones(KEp, dtype=np.float32)

sigma_mx = np.ones(IEp, dtype=np.float32)
sigma_my = np.ones(JEp, dtype=np.float32)
sigma_mz = np.ones(KEp, dtype=np.float32)

##### Grading of PML region #####

rc0 = 1.e-6		# reflection coefficient
gradingOrder = 3.
impedence = np.sqrt(mu_0/epsilon_0)
boundarywidth_x = npml * dx
boundarywidth_y = npml * dy
boundarywidth_z = npml * dz

sigmax_x = -(gradingOrder+1)*np.log(rc0)/(2*impedence*boundarywidth_x)
sigmax_y = -(gradingOrder+1)*np.log(rc0)/(2*impedence*boundarywidth_y)
sigmax_z = -(gradingOrder+1)*np.log(rc0)/(2*impedence*boundarywidth_z)

kappamax_x = 5.
kappamax_y = 5.
kappamax_z = 5.

sigmax_mx = sigmax_x * impedence**2
sigmax_my = sigmax_y * impedence**2
sigmax_mz = sigmax_z * impedence**2

kappamax_mx = kappamax_x
kappamax_my = kappamax_y
kappamax_mz = kappamax_z

space_eps = np.ones((IE,JE,KE),dtype=np.float32) * epsilon_0
space_mu  = np.ones((IE,JE,KE),dtype=np.float32) * mu_0

####################################################################################
################################## APPLYING PML ####################################
####################################################################################

def Apply_PML_3D(**kwargs):

	npml = 10
	dtype = np.float32
	for key, value in kwargs.items():
		if key == 'x':
			x = value

		elif key == 'y':
			y = value

		elif key == 'z':
			z = value

		elif key == 'pml':
			npml = value

		elif key == 'dtype':
			dtype = value

	for i in range(npml):
		if x == '-' :
			
			sigma_x[i]  = sigmax_x * (dtype(npml-i)/npml)**gradingOrder
			kappa_x[i]  = 1 + ((kappamax_x - 1) * ((dtype(npml-i)/npml)**gradingOrder))
			sigma_mx[i] = sigma_x[i] * impedence**2
			kappa_mx[i] = kappa_x[i]

		elif x == '+' :

			sigma_x[-i-1]  = sigmax_x * (dtype(npml-i)/npml)**gradingOrder
			kappa_x[-i-1]  = 1 + ((kappamax_x - 1) * ((dtype(npml-i)/npml)**gradingOrder))
			sigma_mx[-i-1] = sigma_x[-i-1] * impedence**2
			kappa_mx[-i-1] = kappa_x[-i-1]
			
		elif x == '+-' :

			sigma_x[i]  = sigmax_x * (dtype(npml-i)/npml)**gradingOrder
			kappa_x[i]  = 1 + ((kappamax_x-1)*((dtype(npml-i)/npml)**gradingOrder))
			sigma_mx[i] = sigma_x[i] * impedence**2
			kappa_mx[i] = kappa_x[i]

			sigma_x[-i-1]  = sigma_x[i]
			kappa_x[-i-1]  = kappa_x[i]
			sigma_mx[-i-1] = sigma_mx[i]
			kappa_mx[-i-1] = kappa_mx[i]
			
	for j in range(npml):
		if y == '-' :
			
			sigma_y[j]  = sigmax_y * (dtype(npml-j)/npml)**gradingOrder
			kappa_y[j]  = 1 + ((kappamax_y - 1) * ((dtype(npml-j)/npml)**gradingOrder))
			sigma_my[j] = sigma_y[j] * impedence**2
			kappa_my[j] = kappa_y[j]

		elif y == '+' :

			sigma_y[-j-1]  = sigmax_y * (dtype(npml-j)/npml)**gradingOrder
			kappa_y[-j-1]  = 1 + ((kappamax_y - 1) * ((dtype(npml-j)/npml)**gradingOrder))
			sigma_my[-j-1] = sigma_y[-j-1] * impedence**2
			kappa_my[-j-1] = kappa_y[-j-1]
			
		elif y == '+-' :

			sigma_y[j]  = sigmax_y * (dtype(npml-j)/npml)**gradingOrder
			kappa_y[j]  = 1 + ((kappamax_y-1)*((dtype(npml-j)/npml)**gradingOrder))
			sigma_my[j] = sigma_y[j] * impedence**2
			kappa_my[j] = kappa_y[j]

			sigma_y[-j-1]  = sigma_y[j]
			kappa_y[-j-1]  = kappa_y[j]
			sigma_my[-j-1] = sigma_my[j]
			kappa_my[-j-1] = kappa_my[j]

	for k in range(npml):
		if x == '-' :
			
			sigma_x[i]  = sigmax_x * (dtype(npml-i)/npml)**gradingOrder
			kappa_x[i]  = 1 + ((kappamax_x - 1) * ((dtype(npml-i)/npml)**gradingOrder))
			sigma_mx[i] = sigma_x[i] * impedence**2
			kappa_mx[i] = kappa_x[i]

		elif x == '+' :

			sigma_x[-i-1]  = sigmax_x * (dtype(npml-i)/npml)**gradingOrder
			kappa_x[-i-1]  = 1 + ((kappamax_x - 1) * ((dtype(npml-i)/npml)**gradingOrder))
			sigma_mx[-i-1] = sigma_x[-i-1] * impedence**2
			kappa_mx[-i-1] = kappa_x[-i-1]
			
		elif x == '+-' :

			sigma_x[i]  = sigmax_x * (float(npml-i)/npml)**gradingOrder
			kappa_x[i]  = 1 + ((kappamax_x-1)*((float(npml-i)/npml)**gradingOrder))
			sigma_mx[i] = sigma_x[i] * impedence**2
			kappa_mx[i] = kappa_x[i]

			sigma_x[-i-1]  = sigma_x[i]
			kappa_x[-i-1]  = kappa_x[i]
			sigma_mx[-i-1] = sigma_mx[i]
			kappa_mx[-i-1] = kappa_mx[i]

