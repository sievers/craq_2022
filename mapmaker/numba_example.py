import numpy as np
import numba as nb
import time

@nb.njit
def map2tod_numba(x,y,map,data):
    n=len(x)
    for i in np.arange(n):
        data[i]=data[i]+map[x[i],y[i]]


def map2tod(x,y,map,data):
    n=len(x)
    for i in np.arange(n):
        data[i]=data[i]+map[x[i],y[i]]


tmp=np.load('data/tod0.npz')
x=tmp['x']
y=tmp['y']
data=tmp['data']

map=np.zeros([x.max()+1,y.max()+1])
map2=map.copy()

t1=time.time()
map2tod(x,y,map,data)
t2=time.time()
print('took ',t2-t1,' seconds for python loop')

for i in range(5):
    map2[:]=0
    t1=time.time()
    map2tod_numba(x,y,map2,data)
    t2=time.time()
    print('took ',t2-t1,' seconds to do numba on iter ',i)
