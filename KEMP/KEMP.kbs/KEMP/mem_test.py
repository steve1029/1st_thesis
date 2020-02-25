import KEMP
import numpy as np

nx, ny, nz = 1000, 1000, 480

print nx*ny*nz*6*4 / (1024.**3) , 'GiB'

x = np.ones(nx, dtype=np.float32)
y = np.ones(ny, dtype=np.float32)
z = np.ones(nz, dtype=np.float32)
grids = (x, y, z)

fdtd = KEMP.Basic_FDTD('3D', grids, dtype=np.float32, engine='nvidia_cuda', device_id=[0,1])


