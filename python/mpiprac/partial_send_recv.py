from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = MPI.Get_processor_name()

comm.Barrier()

if rank > 0 :

	a = np.arange(1,size)
	b = np.arange(1,size)

	B,A = np.meshgrid(a,b)

	send_data = B[:,rank-1]

	comm.send(send_data, dest=0, tag = rank)
	print "%s: rank %d send" %(hostname,rank), send_data, "to rank 0"

elif rank == 0 :
	
	storage = np.zeros((size-1,size-1))

	recvlist = []

	for tag in range(1,size) :
		recv_data = comm.recv(source=tag, tag=tag)
		recvlist.append(recv_data)
		print "rank 0 received", recv_data, "from rank %d" %(tag)
	
	assert len(recvlist) == (size-1)

	for num, part in enumerate(recvlist):
		storage[num,:] = recvlist[num]	

	assert storage.all() != 0.

	print "rank %d collected data from all nodes and get together in one piece" %rank
	print storage
