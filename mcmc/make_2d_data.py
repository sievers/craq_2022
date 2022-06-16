import numpy as np
from matplotlib import pyplot as plt
import time

plt.ion()
npix=2000
k=np.arange(npix)
k[k>npix/2]=k[k>npix/2]-npix

kx=np.repeat([k],npix,axis=0)
ky=kx.transpose()
k=np.sqrt(kx**2+ky**2)

myamp=1/(1.0+k)**2.0

#A simple way of generating Gaussian random #'s with 
#appropriate phase relations to be the transform of a real
#field is to take the transform of a real field

t1=time.time()
map=np.random.randn(npix,npix)
mapft=np.fft.fft2(map)
t2=time.time()
t_fft=t2-t1

#of course, we could also generate data with the appropriate symmetry, where
#F(kx,ky)=F(-kx,-ky)^*, with special care taking along kx=0,ky=0 axes
t1=time.time()
mapft2_r=np.random.randn(npix,npix)
mapft2_r[1:,0]=mapft2_r[1:,0]+np.flipud(mapft2_r[1:,0])
mapft2_r[0,1:]=mapft2_r[0,1:]+np.flipud(mapft2_r[0,1:])
mapft2_r[1:,1:]=mapft2_r[1:,1:]+np.flipud(np.fliplr(mapft2_r[1:,1:]))
mapft2_i=np.random.randn(npix,npix)
mapft2_i[1:,0]=mapft2_i[1:,0]-np.flipud(mapft2_i[1:,0])
mapft2_i[0,1:]=mapft2_i[0,1:]-np.flipud(mapft2_i[0,1:])
mapft2_i[1:,1:]=mapft2_i[1:,1:]-np.flipud(np.fliplr(mapft2_i[1:,1:]))
mapft2_i[0,0]=0
mapft2=mapft2_r+1J*mapft2_i
t2=time.time()
t_direct=t2-t1
print('time to simulate is ',t_fft,' for fft and ',t_direct,' for direct creation of complex data')

map_back=np.fft.ifft2(mapft*myamp)
map_back=np.real(map_back)
map_back2=np.real(np.fft.ifft2(mapft2*myamp))
