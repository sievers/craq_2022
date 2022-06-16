import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def myfun(pars,x):
    y=np.sin(x*pars[0])+pars[1]*x**2
    return y

def chi_fun(fun,pars,x,data,noise=None):
    y=fun(pars,x)
    resid=data-y
    if noise is None:
        return np.sum(resid**2)
    else:
        return np.sum((resid/noise)**2)

def mcmc(x,data,pars_guess,pars_step,fun,noise=None,nstep=100000):
    npar=len(pars)
    pcur=pars_guess.copy()
    chain=np.zeros([nstep,npar])
    chi_cur=chi_fun(fun,pcur,x,data,noise)
    for i in range(nstep):
        ptrial=pcur+np.random.randn(npar)*pars_step
        chi_trial=chi_fun(fun,ptrial,x,data,noise)
        accept_prob=np.exp(0.5*(chi_cur-chi_trial))
        if np.random.rand(1)<accept_prob:
            chi_cur=chi_trial
            pcur=ptrial
        chain[i,:]=pcur
    return chain


pars=np.asarray([3,.5])
x=np.linspace(-2,2,1000)
y_true=myfun(pars,x)
noise=0.5
y=y_true+np.random.randn(len(y_true))*noise

par_step=np.asarray([0.01,0.01])/10
par_start=np.asarray([2,1])

mychain=mcmc(x,y,par_start,par_step,myfun,0.5)
