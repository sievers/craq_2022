import numpy as np
#import scipy
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import copy
import cg_wplot as cg

try:
    import numba as nb
    have_numba=True
except:
    have_numba=False

#numba version of map2tod.  much faster than
#equivalen python loop
import numba as nb
@nb.njit
def map2tod_numba(x,y,map,data):
    n=len(x)
    for i in np.arange(n):
        data[i]=data[i]+map[x[i],y[i]]

#numba version of tod2map.
@nb.njit
def tod2map_numba(x,y,map,data):
    n=len(x)
    for i in np.arange(n):
        map[x[i],y[i]]=map[x[i],y[i]]+data[i]


def noise_like(pars,noiseft,full=False):
    '''Return the likelihood of the Fourier transform of a set of data.  Pars should be
    a 3-element array with the white level, the knee, and the low-k power law index.
    if you add full=True, we will return the noise model as well.'''
    k=np.arange(len(noiseft))+0.5
    white=pars[0]*len(noiseft)
    knee=pars[1]
    ind=pars[2]
    pred=white*(1+(k/knee)**ind)
    #print(pred[0])
    if pred.min()<=0:
        return 1e20
    num=np.sum(np.abs(noiseft)**2/pred)
    denom=2*np.sum(np.log(pred)) #factor of 2 is for real+im
    like=(num+denom)/len(noiseft)
    if full:
        return like,pred
    else:
        return like


class PowlawWhite:
    '''This is a class to handle noise if the variance is equal to a power law plus a white component.
    __init__ will try to fit the noise model, and apply_noise will return an inverse-variance-weighted
    real-space timestream given an input real-space timestream.'''
    def __init__(self,data,guess,tol=1e-3):
        #print('starting params are ',guess)
        dataft=np.fft.rfft(data)
        self.n=len(data)
        stuff=minimize(noise_like,guess,dataft,tol=tol)
        if stuff.success:
            self.params=stuff.x.copy()
        else:
            self.params=None
    def apply_noise(self,data):
        if self.params is None:
            print("Error in apply noise - we do not have a valid model saved.")
            return None
        dataft=np.fft.rfft(data)
        k=np.arange(len(dataft))+0.5
        white=self.params[0]
        knee=self.params[1]
        ind=self.params[2]
        pred=white*(1+(k/knee)**ind)
        dataft_filt=dataft/pred
        return np.fft.irfft(dataft_filt,self.n)


class Tod:
    '''Class to handle individual chunks of data.  init saves the pointing (x,y) and the data.
    set_noise will let you set the noise model.  For now, use PowlawWhite, but you could
    absolutely use your own noise model.  It just needs to support apply_noise.'''
    def __init__(self,x,y,data):
        self.x=x.copy()
        self.y=y.copy()
        self.data=data.copy()
        self.ndata=len(self.data)
        self.noise=None
        assert(len(self.x)==self.ndata)
        assert(len(self.y)==self.ndata)
    def get_data(self):
        '''Return the data.  It looks silly to have this as a function right now,
        but in real life this routine can look very much more complicated.'''
        return self.data
    def tod2map(self,map,data=None):
        '''Carry out A^T d, use data in data, put answer in map'''
        if data is None: #we'll give ourselves the option of using different data
            data=self.get_data()
        if have_numba:
            tod2map_numba(self.x,self.y,map,data)
        else:
            for i in range(self.ndata):
                map[self.x[i],self.y[i]]=map[self.x[i],self.y[i]]+data[i]

    def map2tod(self,map,data=None):
        '''Carry out Am, use map in map, put answer in data'''
        if data is None:
            data=self.get_data()
        if have_numba:
            map2tod_numba(self.x,self.y,map,data)
        else:
            for i in range(self.ndata):
                data[i]=data[i]+map[self.x[i],self.y[i]]
    def set_noise(self,data=None,noise_type=PowlawWhite,*args,**kwargs):
        '''Set the noise model, using class noise_type'''
        if data is None:
            data=self.get_data()
        self.noise=noise_type(data,*args,**kwargs)
    def apply_noise(self,data=None):
        '''apply the noise model in self.noise to the data.'''
        if self.noise is None:
            print('no valid noise model in tod')
            return None
        else:
            if data is None:
                data=self.get_data()
            return self.noise.apply_noise(data)
        
class TodVec:
    '''Class to handle lots of data chunks (tods).  Saves having to write lots of 
    loops in the main code.'''
    def __init__(self,tods=None):
        if tods is None:
            self.ntod=0
            self.tods=[]
        else:
            self.ntod=len(tods)
            self.tods=[None]*self.ntod
            for i in range(self.ntod):
                self.tods[i]=tods[i]
    def add_tod(self,tod):
        self.ntod=self.ntod+1
        self.tods.append(tod)
        
    def __matmul__(self,mapset):
        '''Carry out A^T N^-1 A m for mapset m.  We'll put the answer in out.
        loop over all the TODs we own, and project the incoming mapset into the data, 
        noise filter the data, then project the noise-filtered data back onto a mapset.'''
        out=mapset.copy()
        out.clear()
        for tod in self.tods:
            tmp=0*tod.get_data() #this is a sloppy way of getting an empty array the size of the data
            #print('mapset is ',type(mapset))
            mapset.mapset2tod(tod,tmp)  #the mapset should know how to project all its pieces into the data
            tmp=tod.apply_noise(tmp)    #noise-filter the data.
            out.tod2mapset(tod,tmp)   #and project the noise-filtered data back onto the output mapset.
        return out
    
class Map:
    '''Class to hold a single map of the sky, in a 2-D pixellated array.'''
    def __init__(self,map):
        self.map=map
    def copy(self):
        return copy.deepcopy(self)
    def clear(self):
        self.map[:]=0
    def map2tod(self,tod,out=None):
        '''The map should be the thing that knows how to turn a TOD into a map and vice-versa.
        That way, if we want a new part of our solution, the new solution is responsible for knowing
        how to go from it to a TOD and back again.'''
        if out is None:
            out=tod.get_data()

        #This line is where the map is responsible for knowing that the map can be turned into a
        #TOD by tod.map2tod.
        tod.map2tod(self.map,out)  
      
    def tod2map(self,tod,data=None):
        '''Same as map2tod, except we project a tod into a map (A^T d)'''
        if data is None:
            data=tod.get_data()
        tod.tod2map(self.map,data)

    #we are now going to overload a bunch of simple operations (adding two maps,
    #multipling by a scalar etc.) so we can treat a map as the linear algebra vector it
    #actually is.  These routines are quite generic, and if you decide you want to use
    #a different sort of map, you could likely use these as-is.  I would suggest doing so
    #via inheritance, and just set up the new init, tod2map and map2tod you would need.
    #(nb - copy and clear are probably also fine to inherit)
    def __matmul__(self,other):
        '''return the dot product of a map with another map.'''
        return np.sum(self.map*other.map)
    def __add__(self,other):
        out=self.copy()
        out.map=self.map+other.map
        return out
    def __mul__(self,val):
        out=self.copy()
        out.map=self.map*val
        return out
    def __rmul__(self,val):
        return self*val
    def __sub__(self,other):
        out=self.copy()
        out.map=self.map-other.map
        return out
    
class Mapset:
    '''Class to handle a bunch of maps.  This looks silly in our simple implementation since
    we only will ever have a single map of the sky.  However, life can easily get more
    complicated and having the ability to transparently handle more thing in your mode.
    can be *extremely* useful.'''
    def __init__(self,maps=None):
        '''Deep down, we're just a list of maps.  Send in the list of maps and you'll have a mapset!'''
        if maps is None:
            self.nmap=0
            self.maps=[]
        else:
            self.nmap=len(maps)
            self.maps=[None]*self.nmap
            for i in range(self.nmap):
                self.maps[i]=maps[i]
    def add_map(self,map):
        '''Add a map to the mapset if we decide we want to.'''
        self.nmap=self.nmap+1
        self.maps.append(map)
    def clear(self):
        for map in self.maps:
            map.clear()
    def copy(self):
        return copy.deepcopy(self)

    def mapset2tod(self,tod,out=None):
        '''Loop over maps for a single TOD to project all of the maps
        we have into the TOD. Save the projected TOD data in out.'''
        if out is None:
            out=0*tod.get_data()
        for i in range(self.nmap):
            self.maps[i].map2tod(tod,out)
        return out
    def tod2mapset(self,tod,data=None):
        '''Take a TOD and project it into all maps in the mapset. '''
        if data is None:
            data=tod.get_data()
        for i in range(self.nmap):
            self.maps[i].tod2map(tod,data)
    def make_rhs(self,todvec):
        '''We also need to make A^T N^-1 d.  For reasons of simpliicity, we're going to stick it here.'''
        out=self.copy()
        out.clear()
        for tod in todvec.tods:
            dat=tod.get_data()
            dat_filt=tod.apply_noise(dat)
            for map in out.maps:
                map.tod2map(tod,dat_filt)
        return out
    #now overload our operators so we can pretend that a mapset is just a vector
    #(which it truly is!).  This lets us use the same conjugate-gradient solver we would
    #for explicit dense matrices/vectors
    def __matmul__(self,other):
        tot=0
        for i in range(self.nmap):
            tot=tot+self.maps[i]@other.maps[i]
        return tot
    def __add__(self,other):
        out=self.copy()
        for i in range(out.nmap):
            out.maps[i]=out.maps[i]+other.maps[i]
        return out
    def __mul__(self,val):
        out=self.copy()
        for i in range(self.nmap):
            out.maps[i]=self.maps[i]*val
        return out
    def __rmul__(self,val):
        return self*val
    def __div__(self,val):
        return self*(1/val)
    def __sub__(self,other):
        out=self.copy()
        for i in range(out.nmap):
            out.maps[i]=out.maps[i]-other.maps[i]
        return out


#main program starts here

#first, read in the data
todvec=TodVec()
ntod=2 #I happened to make 2 TODs for you to look at
for i in range(ntod):
    tmp=np.load('data/tod'+repr(i)+'.npz')
    tod=Tod(tmp['x'],tmp['y'],tmp['data'])
    todvec.add_tod(tod)


#calculate the limits of the TODs since we don't
#necessarily know how large our map needs to be
#if you haven't done much in the way of python list
#comprehensions, you might find the below rather opaque,
#but I hope you will someday find it useful!
xmax=np.max([max(tod.x) for tod in todvec.tods])
ymax=np.max([max(tod.y) for tod in todvec.tods])

#now set up our mapset with an empty map
tmp=np.zeros([xmax+1,ymax+1])
map=Map(tmp)
mapset=Mapset([map])

#now we need to fit the noise for our TODs
#I happen to know what good starting guesses are,
#but in real life you may need to put in some
#effort here.
fknee=10
ind=-2.0
for tod in todvec.tods:
    white_guess=(np.mean(np.abs(np.diff(tod.data))))**2
    knee_guess=len(tod.data)/fknee/2
    ind_guess=ind
    pars_guess=np.asarray([white_guess,knee_guess,ind_guess])
    tod.set_noise(guess=pars_guess*(1+0.01*np.random.randn(len(pars_guess))))
    #if this is ever None, that means our noise fit failed and you
    #should try to patch things up.
    print('noise params are ',tod.noise.params)


#we'll start by making a naive map
mapset_raw=mapset.copy()
mapset_raw.clear()
for tod in todvec.tods:
    mapset_raw.tod2mapset(tod)

plt.ion()
plt.clf()
plt.imshow(mapset_raw.maps[0].map)
plt.show()
    
rhs=mapset.make_rhs(todvec)
x0=0*rhs
x=cg.cg(x0,rhs,todvec,niter=100,plot=False)
