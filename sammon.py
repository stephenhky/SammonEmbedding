import numpy as np
import theano
import theano.tensor as T

# define variables
td = T.iscalar('td')         # target dimensions
mf = T.dscalar('mf')         # magic factor / learning rate

# coordinate variables
Xmatrix = T.dmatrix('Xmatrix')
Ymatrix = T.dmatrix('Ymatrix')

# number of points and dimensions
N, d = Xmatrix.shape

# distance function (Euclidean distance)
dist = lambda i, j: T.sqrt(T.sqr(Xmatrix[i]-Xmatrix[j]))
tdist = lambda i, j: T.sqrt(T.sqr(Ymatrix[i]-Ymatrix[j]))

# cost function
c = T.sum(theano.map(lambda j: T.sum(theano.map(lambda i: dist(i, j), T.arange(j))[0]), T.arange(N))[0])
E = T.sum(theano.map(lambda j: T.sum(theano.map(lambda i: T.sqr(dist(i, j)-tdist(i, j))/dist(i,j),
                                                T.arange(j))[0]),
                     T.arange(N))[0]
          )
E = E / c

# Efcn = theano.function([Xmatrix, Ymatrix], E)

# gradient and second derivatives (not Hessian matrix)
gradE = T.grad(E, Ymatrix)
divgradE = theano.map(lambda i, j: T.grad(gradE[i, j], Ymatrix[i, j]), T.arange(N), T.arange(td))

# update routine
updated_Ymatrix = Ymatrix - mf * gradE / divgradE
updatefcn = theano.funcion([Xmatrix, Ymatrix, mf], updated_Ymatrix)