from mpi4py import MPI
import multiprocessing as mp
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
hostname = MPI.Get_processor_name()
import time
from datetime import datetime as dt

def send(comm,_to,what,tag):

	time.sleep(2)
	comm.send(what, dest=_to, tag=tag)

	pid = mp.current_process().pid
	print("rank %d: I send!, pid: %d" %(rank,pid))
	return

def recv(comm,_from,result,tag):

	recv = comm.recv(source=_from,tag=tag)
	result.put(recv)

	pid = mp.current_process().pid
	print("rank %d: I recv!, pid: %d" %(rank,pid))
	return

if rank == 0:
	
	proc = mp.Process(target=send,args=(comm,1,"Is anybody there?",1))
	proc.start()
	#proc.join()

elif rank == 1:
	t0 = dt.now()
	result = mp.Queue()
	proc = mp.Process(target=recv,args=(comm,0,result,1))
	proc.start()
	#print("haha")
	#proc.join()
	
	a = result.get()
	#b = result.get()
	print("rank %d received from rank 0: " %(rank),a)
	print(dt.now()-t0)
	#print("rank %d received from rank 0: " %(rank),b)
