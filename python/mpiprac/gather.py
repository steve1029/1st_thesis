from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank > 0:
	data = np.arange(rank+1)
elif rank == 0 :
	data = []
gathered_data = comm.gather(data, root=0)

if rank == 0 :
	print gathered_data
	print type(gathered_data)
