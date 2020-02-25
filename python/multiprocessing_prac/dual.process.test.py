import numpy as np
import multiprocessing as mp
import time

def do_work(START, END, storage, timer):

	t0 = time.time()
	
	sum = 0
	
	for i in range(START, END):

		sum += i
		array[i] = sum

	t1 = time.time()

	print("Process for-loop finished within %.2f(s)" %(t1-t0))

	t0 = time.time()
	storage.put(array)
	timer.put(t0)

	print("Queue put finished %.2f(s)" %(time.time()-t0))
	return

if __name__ == '__main__':

	storage1 = mp.Queue()
	storage2 = mp.Queue()

	timer1 = mp.Queue()
	timer2 = mp.Queue()

	print("Test start")

	length = 10**6
	array = np.zeros(length,dtype=complex)

	Process1 = mp.Process(target=do_work, args=(0,length/2,storage1,timer1))
	Process2 = mp.Process(target=do_work, args=(length/2,length,storage2,timer2))

	Process1.start()
	Process2.start()

	result1 = storage1.get()
	result2 = storage2.get()

	t1 = time.time() - timer1.get()
	t2 = time.time() - timer2.get()

	print("Communication time of Process1 %.2f(s) and Process2 %.2f(s)" %(t1, t2))

	Process1.join()
	Process2.join()

	array[0:length/2] = result1[0:length/2]
	array[length/2:length] = result2[length/2:length]
	
	print("Test finished")
