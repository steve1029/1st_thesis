import numpy as np
from mpi4py import MPI
from datetime import datetime as dtm
from scipy.optimize import curve_fit
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def linear_profile(x, a, b):
    return a*x + b

# Blocking Communications
def exchange_block(size):
    if   rank == 0:
        data0 = np.arange(size, dtype=np.float32)
        data1 = np.empty(size, dtype=np.float32)
        t0 = dtm.now()
        comm.Send(data0, dest=1, tag=98)
        comm.Recv(data1, source=1, tag=99)
        t1 = dtm.now()
    elif rank == 1:
        data0 = np.empty(size, dtype=np.float32)
        data1 = np.arange(size, dtype=np.float32)
        t0 = dtm.now()
        comm.Recv(data0, source=0, tag=98)
        comm.Send(data1, dest=0, tag=99)
        t1 = dtm.now()
    res = abs(data1 - data0).sum()
    dt = t1 - t0
    print rank, res, dt
    return dt.total_seconds()

# Nonblocking Communications
def exchange_nonblock(size):
    if   rank == 0:
        data0 = np.arange(size, dtype=np.float32)
        data1 = np.empty(size, dtype=np.float32)
        t0 = dtm.now()
        req_send = comm.Isend(data0, dest=1, tag=98)
        req_recv = comm.Irecv(data1, source=1, tag=99)
    elif rank == 1:
        data0 = np.empty(size, dtype=np.float32)
        data1 = np.arange(size, dtype=np.float32)
        t0 = dtm.now()
        req_send = comm.Isend(data1, dest=0, tag=99)
        req_recv = comm.Irecv(data0, source=0, tag=98)

    req_send.Wait()
    req_recv.Wait()

    t1 = dtm.now()
    res = abs(data1 - data0).sum()
    dt = t1 - t0
    print rank, res, dt
    return dt.total_seconds()

# Nonblocking Communications: scheduling
def exchange_nonblock_schedule(size):
    if   rank == 0:
        data0 = np.arange(size, dtype=np.float32)
        data1 = np.empty(size, dtype=np.float32)
        req_send = comm.Send_init([data0, MPI.FLOAT], 1, tag=0)
        req_recv = comm.Recv_init([data1, MPI.FLOAT], 1, tag=1)
#        req_send = comm.Send_init(data0, 1, tag=0)
#        req_recv = comm.Recv_init(data1, 1, tag=1)
    elif rank == 1:
        data0 = np.empty(size, dtype=np.float32)
        data1 = np.arange(size, dtype=np.float32)
        req_send = comm.Send_init([data1, MPI.FLOAT], 0, tag=1)
        req_recv = comm.Recv_init([data0, MPI.FLOAT], 0, tag=0)
#        req_send = comm.Send_init(data1, 0, tag=1)
#        req_recv = comm.Recv_init(data0, 0, tag=0)

    comm.Barrier()
    t0 = dtm.now()
    req_send.Start()
    req_recv.Start()
    req_recv.Wait()
    req_send.Wait()
    comm.Barrier()
    t1 = dtm.now()
    res = abs(data1 - data0).sum()
    dt = t1 - t0

#    print rank, res, dt
    return dt.total_seconds()

#n_size = 2000000

#exchange_block(n_size)
#exchange_nonblock(n_size)
#exchange_nonblock_schedule(n_size)

sizes  =   np.arange(1000000, 10000000, 1000000)
nbytes = 4*np.arange(1000000, 10000000, 1000000)
times  = np.zeros(sizes.size, dtype=np.float32)
for i, s in enumerate(sizes):
    dt = exchange_nonblock_schedule(s)
    times[i] = dt
    if rank == 0: print dt, nbytes[i]/dt

popt, pcov = curve_fit(linear_profile, nbytes, times)

if rank == 0:
    print 1.0e-6/popt[0], 'MBytes/s'
    print popt[1], 's'

    import matplotlib.pyplot as plt
    plt.plot(nbytes, times)
    plt.show()
