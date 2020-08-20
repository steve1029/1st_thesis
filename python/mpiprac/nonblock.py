from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

if rank == 0:
	data = np.arange(10, dtype=np.complex128)
	print(rank, data)
	req = comm.Isend(data, dest=1, tag=1)
	req.Wait()

elif rank == 1:
	data = np.zeros(10, dtype=np.complex128)
	req = comm.Irecv(data, source=0, tag=1)
	req.Wait()
	print(data)
	print(data)
