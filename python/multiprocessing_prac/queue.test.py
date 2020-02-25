import multiprocessing as mp
import time, os

def do_work(Q,sleeptime):
	
	time.sleep(sleeptime)
	recv = Q.get()
	print(os.getpid(),"I received:",recv)

if __name__ == '__main__':
	
	Q = mp.Queue()
	p1 = mp.Process(target=do_work, args=(Q,2))
	p2 = mp.Process(target=do_work, args=(Q,3))

	p1.start()
	p2.start()

	Q.put("Is anybody there?1")
	time.sleep(5)
	Q.put("Is anybody there?2")

	print(os.getpid(), "I send!")

	p1.join()
