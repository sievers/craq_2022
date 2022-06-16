import numpy as np
from matplotlib import pyplot as plt


#we are going to pick a random function.  Others are welcome!
def ourfun(x,pars):
    #y=a sin(b/(x-c))+d
    y=pars[0]*np.sin(pars[1]/(x-pars[2]))+pars[3]
    return y

#the mcmc routine will want a function that returns chi^2
def our_chisq(data,pars):
    #we need a function that calculates chi^2 for us for the MCMC
    #routine to call
    x=data[0]
    y=data[1]
    noise=data[2]
    model=ourfun(x,pars)
    chisq=np.sum( (y-model)**2/noise**2)
    return chisq

#work out numerical derivatives of our function.  We need
#to specify a sensible amount to shift the parameters.  For better
#accuracy, take the 2-sided derivative.
def num_deriv(fun,x,pars,dpar):
    #calculate numerical derivatives of 
    #a function for use in e.g. Newton's method or LM
    derivs=np.zeros([len(x),len(pars)])
    for i in range(len(pars)):
        pars2=pars.copy()
        pars2[i]=pars2[i]+dpar[i]
        f_right=fun(x,pars2)
        pars2[i]=pars[i]-dpar[i]
        f_left=fun(x,pars2)
        derivs[:,i]=(f_right-f_left)/(2*dpar[i])
    return derivs


#given a function that returns chi^2, and parameter step sizes
#in par_step, create a Markov chain of fixed length nstep.
#note that we never look inside data, so we could stash the errors
#in there as well, which works as long as what you put in data
#matches what chifun is expecting.
def run_mcmc(pars,data,par_step,chifun,nstep=5000):
    #initialize all the things
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)

    #make sure we know what chi^2 is at our start
    chi_cur=chifun(data,pars)
    for i in range(nstep):
        #take a random trial step, then evaluate chi^2
        pars_trial=pars+np.random.randn(npar)*par_step
        chi_trial=chifun(data,pars_trial)
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            #if we accepted the step, update our current parameters and chi^2.
            #you'll see there's no else - if we don't accept the step, we stay at the
            #current parameters.
            pars=pars_trial
            chi_cur=chi_trial
        #add wherever we ended up (either new step or not) into our chain/chi^2 vector.
        chain[i,:]=pars
        chivec[i]=chi_cur
    return chain,chivec
        
#ok, we'll pick an x range, and some random parameters
#for our function.
x=np.linspace(0.1,5,5000)
pars_true=np.asarray([1,1,0,0],dtype='double')

#we'll cheat a little bit and use the true parameters for
#a start to our fit
pars=pars_true.copy()
y_true=ourfun(x,pars) #might as well save what we sould get
noise=0.1
y=y_true+noise*np.random.randn(len(x)) #and this is our data with noise

#run Newton's with numerical derivatives
Ninv=np.eye(len(x))/noise**2  #this is our noise matrix
dpar=np.ones(len(pars))*1e-2  #and this is how big our steps will be in numerical derivatives
for i in range(10):  #we could be smarter about checking convergence, but for this example we don't need to.
    #this loop runs Newton's method in multi-dimensions.
    #With noise, the linearized solution to finding where
    #the gradient of chi^2=2 is A^T N^-1 A dp = A^T N^-1 r
    #where dp is the shift in parameters, r is the residual between
    #the data and our current model, N is the noise matrix, and A is
    #the matrix of function derivatives with respect to our parameters,
    #evaluated at our current set of parameters.
    #we'll call A^T N^-1 A lhs (for left-hand side)
    #and A^T N^-1 r the rhs (for right-hand side).
    #we then have dp=inv(lhs)*rhs
    model=ourfun(x,pars) #get the model
    derivs=num_deriv(ourfun,x,pars,dpar) #get A
    resid=y-model #residual is true data minus current model
    lhs=derivs.T@Ninv@derivs #make the LHS
    rhs=derivs.T@Ninv@resid  #make the RHS
    lhs_inv=np.linalg.inv(lhs)  #invert the LHS
    step=lhs_inv@rhs  #find our parameter step
    pars=pars+step  #update our parameters
    print(pars)  #you should watch this in the general case.  Newton's method may not be stable!

#we are going to start by running a single chain 
#since we have a curvature estimate from Newton's method, we can
#guess our chain sampling using that
par_sigs=np.sqrt(np.diag(lhs_inv)) #these are our Gaussian parameter errors
data=[x,y,noise]  #this is what the chi^2 function expects
chain,chivec=run_mcmc(pars,data,par_sigs,our_chisq,nstep=20000)
#now that we have a first chain, we can update our estimate
#of the mean/errors by looking at the scatter of the chain
par_sigs=np.std(chain,axis=0)  
par_means=np.mean(chain,axis=0)

#now let's run multiple chains, using our preliminary chain to guide us
#with multiple chains, we can look at the Gelman-Rubin convergence
nchain=4
all_chains=[None]*nchain
for i in range(nchain):
    pars_start=par_means+3*par_sigs*np.random.randn(len(pars))
    chain,chivec=run_mcmc(pars_start,data,par_sigs,our_chisq,nstep=20000)
    all_chains[i]=chain


#just for sanity, let's check for each pair of chains how the difference of the means
#compares to the scatter within the chain

for i in range(nchain):
    for j in range(i+1,nchain):
        mean1=np.mean(all_chains[i],axis=0)
        mean2=np.mean(all_chains[j],axis=0)
        std1=np.std(all_chains[i],axis=0)
        std2=np.std(all_chains[j],axis=0)
        print('param difference in sigma is ',(mean1-mean2)/(0.5*(std1+std2)))

