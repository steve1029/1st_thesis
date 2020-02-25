import multiprocessing as mp
import numpy as np

array = np.zeros(10)

def fill(array, Q):

	filler = np.arange(10)

	for i in range(10):
		array[i] = filler[i]

	Q.put(array)

	return

Q = mp.Queue()
do_fill = mp.Process(target=fill, args=(array,Q))
do_fill.start()

do_fill.join()

print(array)
print(Q.get())
print(array)
