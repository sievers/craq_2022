import numpy as np
import cg


def matgen(n=100,diag=10,maxeig=1):
    '''Make a positive-definite square matrix of size n, with eigenvalues
    between 0 and maxeig, and an additional offset on the diagonal of diag.'''
    mat=np.random.randn(n,n)
    mat=mat+mat.T
    e,v=np.linalg.eigh(mat)
    e=np.random.rand(n)*maxeig

    mm=v@np.diag(e)@v.T
    return mm+np.eye(n)*diag




n=100 #pick a random matrix size
A=matgen(n,diag=3) #generate a positive-definite matrix of this size
b=np.random.randn(n) #pick a random rhs to Ax=b
x0=0*b #this is a starting point - in the absence of a better guess, might as well be zero
x=cg.cg(x0,b,A,niter=10) #try to solve our equation.
print('residual RMS of Ax-b is ',np.std(A@x-b))



