import numpy as np
import theano
import theano.tensor as T

# define variables
# td = T.iscalar('td')         # target dimensions
mf = T.dscalar('mf')         # magic factor / learning rate

# coordinate variables
Xmatrix = T.dmatrix('Xmatrix')
Ymatrix = T.dmatrix('Ymatrix')

# number of points and dimensions
N, d = Xmatrix.shape
_, td = Ymatrix.shape

# distance function (Euclidean distance)
# TODO: precompute it
dist = lambda i, j: T.sqrt(T.sum(T.sqr(Xmatrix[i]-Xmatrix[j])))
tdist = lambda i, j: T.sqrt(T.sum(T.sqr(Ymatrix[i]-Ymatrix[j])))

# cost function
n_grid = T.mgrid[0:N, 0:N]
ni = n_grid[0].flatten()
nj = n_grid[1].flatten()
c_terms, _ = theano.scan(lambda i, j: T.switch(T.lt(i, j), dist(i, j), 0),
                         sequences=[ni, nj])
c = T.sum(c_terms)

s_term, _ = theano.scan(lambda i, j: T.switch(T.lt(i, j),
                                              T.sqr(dist(i, j)-tdist(i, j))/dist(i, j),
                                              0),
                        sequences=[ni, nj])
s = T.sum(s_term)

E = s / c

# gradient
gradE = T.grad(E, Ymatrix)

# second derivatives (not Hessian matrix)
imgrid = T.mgrid[0:N, 0:td]
flattenii = imgrid[0].flatten()
flattenjj = imgrid[1].flatten()
divgradE, divgradupdates = theano.scan(lambda i, j, gE, Y: T.grad(gE[i, j], Y)[i, j],
                                       sequences=[flattenii, flattenjj],
                                       non_sequences=[gradE, Ymatrix])
divgradE = T.reshape(divgradE, (N, td))

# update routine
updated_Ymatrix = Ymatrix - mf * gradE / divgradE
updatefcn = theano.function([Xmatrix, Ymatrix, mf], updated_Ymatrix)