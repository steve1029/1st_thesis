import os, sys, time
import subprocess as sp
import psutil as ps

def ext_exec_wait(cmd):
    out, err = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    if err != '': print err
    return out

def ext_exec_nowait(cmd):
    sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)

cpuinfo = ext_exec_wait('/bin/cat /proc/cpuinfo')

cpu_name = cpuinfo[79:118]

print cpu_name
