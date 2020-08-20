from mpi4py import MPI
import numpy as np
import subprocess as sp
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

n = 1

dtype = np.int64

randint = np.zeros(1,dtype=dtype)

assert randint.dtype == dtype

comm.Barrier()

for i in range(1,size):
    if rank == i:
        randint = np.random.randint(low=0, high=10)
        print("HOST: %s, RANK: %d, RAND: %d" %(hostname, rank, randint))

randint = comm.reduce(randint, op=MPI.SUM, root=0)

comm.Barrier()

if rank == 0:
    print ("HOST: %s, RANK: %d, TOTAL: %d" %(hostname, rank, randint))
