import numpy as np
from matplotlib import pyplot as plt
import camb

def get_spectrum(pars,lmax=3000):
    '''Routine to return the tt power spectrum based on input 6-parmeter cosmological model.
    Input parameters should be in a list/array with entries [H0,ombh^2,omch^2,tau,A_s,n_s]'''
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

def draw_sample(mat,n=1):
    '''routine to draw a sample dataset from a covariance matrix.'''
    rr=np.linalg.cholesky(mat)
    if n==1:
        return rr@np.random.randn(rr.shape[0])
    else:
        tmp=np.random.randn(rr.shape[0],n)
        return rr@tmp

def num_deriv(fun,x,pars,dpar):
    #calculate numerical derivatives of 
    #a function for use in e.g. Newton's method or LM
    
    ff=fun(pars,x)
    derivs=np.zeros([len(ff),len(pars)])
    for i in range(len(pars)):
        #print('i is ',i)
        pars2=pars.copy()
        pars2[i]=pars2[i]+dpar[i]
        f_right=fun(pars2,x)
        pars2[i]=pars[i]-dpar[i]
        f_left=fun(pars2,x)
        derivs[:,i]=(f_right-f_left)/(2*dpar[i])
    return derivs



def run_mcmc(pars,data,par_step,chifun,mask=None,nstep=5000,outfile=None):
    '''Generic routine to run a Markov chain. Likelihood should be
    chifun(data,pars).  Steps are set by par_step.  If it is an array, take
    uncorrelated steps with length set by par_step.  If it is an array, 
    take correlated steps.  If outfile is non-zero, dump to a file as we go along'''


    npar=len(pars)
    if mask is None:
        mask=np.ones(npar,dtype='bool')

    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    chi_cur=chifun(data,pars)
    if not(outfile is None):
        f=open(outfile,'w')
    else:
        f=None
    for i in range(nstep):
        print(i)
        if len(par_step.shape)==1:
            #pars_trial=pars+np.random.randn(npar)*par_step
            pars_trial=pars.copy()
            pars_trial[mask]=pars_trial[mask]+np.random.randn(np.sum(mask))*par_step[mask]
        else:
            pars_trial=pars.copy()
            pars_trial[mask]=pars_trial[mask]+draw_sample(par_step)
        chi_trial=chifun(data,pars_trial)
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            print('accepting')
            pars=pars_trial
            chi_cur=chi_trial
        chain[i,:]=pars
        chivec[i]=chi_cur
        if not(f is None):
            to_print=repr(chi_cur)
            for j in range(len(pars)):
                to_print=to_print+' '+repr(pars[j])
            f.write(to_print+'\n')
            f.flush()
    return chain,chivec

def planck_chisq(data,pars):
    spec=data[0]
    errs=data[1]
    ellmax=len(spec)+200
    model=get_spectrum(pars,ellmax)
    resid=spec-model[:len(spec)]
    return np.sum( (resid/errs)**2)



#read the planck data/errs
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
#as a hack, we'll just take the average of the upper and lower errors for a Gaussian errorbar
errs=0.5*(planck[:,2]+planck[:,3]);
pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
dpar=0.001*pars #this will be the width we use for our numerical derivatives
ellmax=3000

pars_org=pars.copy()

#run Newton's with numerical derivatives
Ninv=np.diag(1/errs**2)
mask=np.asarray([True,True,True,False,True,True])

#my initial guess didn't converge with tau, but this one does.
#pars=np.asarray([6.76446847e+01, 2.22792779e-02, 1.18973378e-01, 5.00000000e-02,2.07465890e-09, 9.69863965e-01])
pars=np.asarray([67, 2.2e-02, 1.2e-01, 8.0e-02,2.2e-09, 0.97])
mask[:]=True #we'll only float parameters where this mask is set to True
converged=False
i=0
imax=10 #max number of Newton's iterations
dchi=0.1 #we'll call ourselves converged when chi^2 changes by less than this
chisq_old=1e20
while (converged==False):
    i=i+1
    model=get_spectrum(pars,ellmax)
    model=model[:len(ell)]
    derivs=num_deriv(get_spectrum,ellmax,pars,dpar)
    derivs=derivs[:len(ell),:]
    resid=spec-model
    chisq=resid@Ninv@resid
    if (np.abs(chisq-chisq_old)<dchi):
        converged=True
    chisq_old=chisq
    lhs=derivs.T@Ninv@derivs
    rhs=derivs.T@Ninv@resid
    #lhs_inv=np.linalg.inv(lhs)
    lhs_use=lhs[mask,:]
    lhs_use=lhs_use[:,mask]
    rhs_use=rhs[mask]
    lhs_inv=np.linalg.inv(lhs_use)
    
    #step=lhs_inv@rhs
    #pars=pars+step
    step=lhs_inv@rhs_use
    pars[mask]=pars[mask]+step
    print(chisq,pars,step)
    if i==imax:
        print('failed to converge in Newton''s method.')
        break


#since we have a curvature estimate from Newton's method, we can
#guess our chain sampling using that
#par_sigs=np.sqrt(np.diag(lhs_inv)) #if we were to try this, you'd see the chain doesn't work.  why?
par_sigs=lhs_inv #this is better...
#data=[x,y,noise]
data=[spec,errs]

chain,chivec=run_mcmc(pars,data,par_sigs,planck_chisq,nstep=1000,outfile='planck_chain.txt')
assert(1==0)

#here's how we could run a chain with tau fixed
mask2=np.asarray([True,True,True,False,True,True])
tmp=lhs[mask2,:]
par_sigs_masked=np.linalg.inv(tmp[:,mask2])
chain2,chivec2=run_mcmc(pars,data,par_sigs_masked,planck_chisq,mask=mask2,nstep=5000,outfile='planck_no_tau.txt')

