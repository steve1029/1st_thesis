import multiprocessing as mp
import numpy as np
import os, time

def do_work(start, end, result):
	sum = 0

	for i in range(start, end):
		sum += i
	
	result.put(sum)
	print("finish")
	return

def with_2_process(START,END):

	t0 = time.time()

	result = mp.Queue()

	process1 = mp.Process(target=do_work, args=(START, int(END/2), result))
	process2 = mp.Process(target=do_work, args=(int(END/2), END  , result))

	process1.start()
	process2.start()
	process1.join()
	process2.join()
	result.put('STOP')

	sum = 0
	while True:
		tmp = result.get()
		if tmp == 'STOP': break
		else: sum += tmp
	
	t1 = time.time() - t0
	print("with 2 cpu, Result: %d, time: %.2f s" %(sum,t1))

	return

def with_4_process(START, END):

	t0 = time.time()

	result = mp.Queue()

	process1 = mp.Process(target=do_work, args=(START,        int(END/4)  , result))
	process2 = mp.Process(target=do_work, args=(int(END/4)  , int(END/2)  , result))
	process3 = mp.Process(target=do_work, args=(int(END/2)  , int(3*END/4), result))
	process4 = mp.Process(target=do_work, args=(int(3*END/4), int(4*END/4), result))

	process1.start()
	process2.start()
	process3.start()
	process4.start()

	process1.join()
	process2.join()
	process3.join()
	process4.join()

	result.put('STOP')

	sum = 0
	while True:
		tmp = result.get()
		if tmp == 'STOP': break
		else: sum += tmp
	t1 = time.time() - t0
	print("with 4 cpu, Result: %d, time: %.2f s" %(sum,t1))

	return

def with_6_process(START, END):

	t0 = time.time()

	result = mp.Queue()

	process1 = mp.Process(target=do_work, args=(START       , int(1*END/6), result))
	process2 = mp.Process(target=do_work, args=(int(1*END/6), int(2*END/6), result))
	process3 = mp.Process(target=do_work, args=(int(2*END/6), int(3*END/6), result))
	process4 = mp.Process(target=do_work, args=(int(3*END/6), int(4*END/6), result))
	process5 = mp.Process(target=do_work, args=(int(4*END/6), int(5*END/6), result))
	process6 = mp.Process(target=do_work, args=(int(5*END/6), int(6*END/6), result))

	process1.start()
	process2.start()
	process3.start()
	process4.start()
	process5.start()
	process6.start()

	process1.join()
	process2.join()
	process3.join()
	process4.join()
	process5.join()
	process6.join()

	result.put('STOP')

	sum = 0
	while True:
		tmp = result.get()
		if tmp == 'STOP': break
		else: sum += tmp
	t1 = time.time() - t0
	print("with 6 cpu, Result: %d, time: %.2f s" %(sum,t1))

	return

def with_8_process(START, END):

	t0 = time.time()

	result = mp.Queue()

	process1 = mp.Process(target=do_work, args=(START       , int(1*END/8), result))
	process2 = mp.Process(target=do_work, args=(int(1*END/8), int(2*END/8), result))
	process3 = mp.Process(target=do_work, args=(int(2*END/8), int(3*END/8), result))
	process4 = mp.Process(target=do_work, args=(int(3*END/8), int(4*END/8), result))
	process5 = mp.Process(target=do_work, args=(int(4*END/8), int(5*END/8), result))
	process6 = mp.Process(target=do_work, args=(int(5*END/8), int(6*END/8), result))
	process7 = mp.Process(target=do_work, args=(int(6*END/8), int(7*END/8), result))
	process8 = mp.Process(target=do_work, args=(int(7*END/8), int(8*END/8), result))

	process1.start()
	process2.start()
	process3.start()
	process4.start()
	process5.start()
	process6.start()
	process7.start()
	process8.start()

	process1.join()
	process2.join()
	process3.join()
	process4.join()
	process5.join()
	process6.join()
	process7.join()
	process8.join()

	result.put('STOP')

	sum = 0
	while True:
		tmp = result.get()
		if tmp == 'STOP': break
		else: sum += tmp
	
	t1 = time.time() - t0
	print("with 8 cpu, Result: %d, time: %.2f s" %(sum,t1))

	return

def with_n_process(n, START, END):
	
	t0 = time.time()
	result = mp.Queue()
	portion = np.zeros(n+1,dtype=int)

	for i in range(1,n+1):
		portion[i] = int(START + ((END-START) * np.float64(i)/n))

	assert portion[-1] == END, "(END-START)/n is not integer."

	procs = []
	for i in range(n):
		start = portion[i  ] 
		end   = portion[i+1]
		proc = mp.Process(target=do_work,args=(start,end,result))
		procs.append(proc)
		proc.start()

	print("can i?")
	
	for i in range(n): procs[i].join()
	result.put('STOP')

	sum = 0

	while True:
		tmp = result.get()
		if tmp == 'STOP': break
		else: sum += tmp
	
	t1 = time.time() - t0

	print("with %d cpu, Result: %d, time: %.2f s" %(n,sum,t1))

	return

if __name__ == "__main__":

	print("total number of cpu: %d" %(mp.cpu_count()))

	START, END = 0, 48 * 10**(7)
#	with_2_process(START,END)
#	with_4_process(START,END)
#	with_6_process(START,END)
#	with_8_process(START,END)

	#with_n_process(2, START, END)
	with_n_process(4, START, END)
	#with_n_process(6, START, END)
	#with_n_process(8, START, END)
