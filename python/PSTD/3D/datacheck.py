import os

def makefolder(directory):
	
	try :
		os.mkdir(directory)

	except Exception as err :
		return "%s Directory already exists" %directory

	return "%s Directory is created" %directory

def datacheck(data,directory):
	"""Save 3D array as text file. Written to compensate
	np.savetxt cause it is not suitable for saving 3d array data.
	
	PARAMETERS
	-----------
	data : arr_like
		3d numpy array or list, anyting kind of.
	
	directory : string
		directory to save
	
	RETURN
	-----------
	None
	"""

	X = data.shape[0]
	Y = data.shape[1]
	Z = data.shape[2]

	f = open('%s' %(directory),'w',newline='\r\n')

	for x in range(X):
		for y in range(Y):
			for z in range(Z):
				f.write("%.3g\t" %data[x,y,z].real)
			f.write('\n')
		f.write("\n\n\n")
	
	f.close()

	return None
