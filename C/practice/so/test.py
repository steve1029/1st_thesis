from ctypes import *
import os

s='Hello world?'

clib = cdll.LoadLibrary('libc.so.6')
print 'clib.time() = %d' % clib.time(None)
clib.printf("clib.printf(s) = <%s>\n", s)

mylib = cdll.LoadLibrary('%s/libmylib.so' % os.getcwd())

print 'mylib.sum(1,2) => %d' % mylib.sum(1,2)

print 'mystrlen("%s") => %d' % (s,mylib.mystrlen(s))

v=[]
v.append('a')
v.append('0')
v.append('1')
v.append(chr(0))
v.append('2')
v.append('3')
s = ''.join(v)
print 'python s = <%s>' % s
r = mylib.hexdump(s,len(s))
print "mylib.hexdump(s,len(s)) = %s" % r
