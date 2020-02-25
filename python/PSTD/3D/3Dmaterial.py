import numpy as np
from scipy.constants import c, epsilon_0, mu_0

def rectangle_3D(backend_lower_left,front_upper_right, eps_r, mu_r, conductivity):
	"""Put 3D Cubic in main space.

	PARAMETERS
	--------------
	backend_lower_left : int tuple
		coordinates of backend_lower_left

	front_upper_right : int tuple
		coordinates of front_upper_right

	eps_r : float or tuple
		relative epsilon of rectanguler slab
		anisotropic magnetic materials can be applied by tuple.

	mu_r : float or tuple
		relative epsilon of rectanguler slab
		anisotropic dielectric materials can be applied by tuple.
		
	
	Returns
	-------------
	None
	"""
	dtype = np.float32

	if type(eps_r) == int:
		eps_r = dtype(eps_r)
	elif type(eps_r) == dtype:
		eps_rx = eps_r
		eps_ry = eps_r
		eps_rz = eps_r
	elif type(eps_r) == tuple:
		eps_rx = eps_r[0]
		eps_ry = eps_r[1]
		eps_rz = eps_r[2]

	if type(mu_r) == dtype:
		mu_rx = mu_r
		mu_ry = mu_r
		mu_rz = mu_r
	elif type(mu_r) == int:
		mu_rx = dtype(mu_r)
		mu_ry = dtype(mu_r)
	elif type(mu_r) == tuple:
		mu_rx = mu_r[0]
		mu_ry = mu_r[1]
		mu_rz = mu_r[2]

	if type(conductivity) == int:
		conductivity_x = dtype(conductivity)
		conductivity_y = dtype(conductivity)
		conductivity_z = dtype(conductivity)

	elif type(conductivity) == dtype:
		conductivity_x = conductivity
		conductivity_y = conductivity
		conductivity_z = conductivity

	elif type(conductivity) == tuple:
		conductivity_x = conductivity[0] / eps_rx
		conductivity_y = conductivity[1] / eps_ry
		conductivity_z = conductivity[2] / eps_rz

	xcoor = backend_lower_left[0]
	ycoor = backend_lower_left[1]
	zcoor = backend_lower_left[2]
	
	height = front_upper_right[0] - backend_lower_left[0]
	depth  = front_upper_right[1] - backend_lower_left[1]
	width  = front_upper_right[2] - backend_lower_left[2]

	for i in range(height):
		for j in range(depth):
			for k in range(width):
				space_eps[xcoor+i,ycoor+j,zcoor+k] = eps_r * epsilon_0
				space_mu [xcoor+i,ycoor+j,zcoor+k] = mu_r  * mu_0

	return None
