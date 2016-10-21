import numpy as np
import theano
import theano.tensor as T

import numerical_gradients as ng

# define variables
mf = T.dscalar('mf')         # magic factor / learning rate

# coordinate variables
Xmatrix = T.dmatrix('Xmatrix')
Ymatrix = T.dmatrix('Ymatrix')

# number of points and dimensions (user specify them)
N, d = Xmatrix.shape
_, td = Ymatrix.shape

# grid indices
n_grid = T.mgrid[0:N, 0:N]
ni = n_grid[0].flatten()
nj = n_grid[1].flatten()

# distance function (Euclidean distance)
# dist = lambda i, j: T.sqrt(T.sum(T.sqr(Xmatrix[i]-Xmatrix[j])))
# tdist = lambda i, j: T.sqrt(T.sum(T.sqr(Ymatrix[i]-Ymatrix[j])))

# cost function
c_terms, _ = theano.scan(lambda i, j: T.switch(T.lt(i, j),
                                               T.sqrt(T.sum(T.sqr(Xmatrix[i]-Xmatrix[j]))),
                                               0),
                         sequences=[ni, nj])
c = T.sum(c_terms)

s_term, _ = theano.scan(lambda i, j: T.switch(T.lt(i, j),
                                              T.sqr(T.sqrt(T.sum(T.sqr(Xmatrix[i]-Xmatrix[j])))-T.sqrt(T.sum(T.sqr(Ymatrix[i]-Ymatrix[j]))))/T.sqrt(T.sum(T.sqr(Xmatrix[i]-Xmatrix[j]))),
                                              0),
                        sequences=[ni, nj])
s = T.sum(s_term)

E = s / c

# function compilation and optimization
Efcn = theano.function([Xmatrix, Ymatrix], E)

# training
def sammon_embedding(Xmat, initYmat, alpha=0.3, tol=1e-8):
    N, d = Xmat.shape
    NY, td = initYmat.shape
    if N != NY:
        raise ValueError('Number of vectors in Ymat ('+str(NY)+') is not the same as Xmat ('+str(N)+')!')

    Efcn_X = lambda Ymat: Efcn(Xmat, Ymat)
    # iteration
    oldYmat = initYmat
    converged = False
    while not converged:
        newYmat = oldYmat - alpha*ng.tensor_gradient(Efcn_X, oldYmat, tol=tol)/ng.tensor_divgrad(Efcn_X, oldYmat, tol=tol)
        if np.all(np.abs(Efcn_X(newYmat)-Efcn_X(oldYmat))<tol):
            converged = True
        oldYmat = newYmat
    return oldYmat