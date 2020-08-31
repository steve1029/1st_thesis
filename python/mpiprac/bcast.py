from mpi4py import MPI
import numpy as np
import subprocess as sp
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

#print "HOST: %s, RANK: %d, of %d nodes" % (hostname, rank, size)

n = 1

dtype = np.int32

comm.Barrier()

if rank == 0:
	randint = np.random.randint(low=0,high=100, size=n)
	print("Root: %s, number: %d" %(hostname,randint))
else :
	randint = np.zeros(n,dtype=dtype)

comm.Barrier()
randint = comm.bcast(randint, root=0)

if   rank == 1: print("Host: %s, rank: %d, recv: %d" %(hostname,rank,randint))
elif rank == 2:	print("Host: %s, rank: %d, recv: %d" %(hostname,rank,randint))
elif rank == 3:	print("Host: %s, rank: %d, recv: %d" %(hostname,rank,randint))

