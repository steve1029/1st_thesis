import os
import subprocess as sp

homename = 'yhome%01d'
nodename = 'y%03d'

homes = []
nodes = []

for i in range(1,4):
	homes.append(homename %i)

for i in range(111,119):
	nodes.append(nodename %i)

f = open("/root/3D_PSTD/note.py",'a')

f.write('writting test1')
f.close()
