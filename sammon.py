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
N, d = T.shape(Xmatrix)

# distance function (Euclidean distance)
dist = lambda i, j: T.sqrt(T.sqr(Xmatrix[i]-Xmatrix[j]))
tdist = lambda i, j: T.sqrt(T.sqr(Ymatrix[i]-Ymatrix[j]))

# cost function
c = theano.reduce(T.add,
                  theano.map(lambda j: theano.reduce(T.add,
                                                     theano.map(lambda i: dist(i, j),
                                                                T.arange(j))
                                                     ),
                             T.arange(N))
                  )
E = theano.reduce(T.add,
                  theano.map(lambda j: theano.reduce(T.add,
                                                     theano.map(lambda i: T.sqr(dist(i, j)-tdist(i, j))/dist(i,j),
                                                                T.arange(j)),
                                                     ),
                             T.arange(N))
                  )
E = E / c

# gradient and second derivatives (not Hessian matrix)
gradE = T.grad(E, Ymatrix)
divgradE = theano.map(lambda i, j: T.grad(gradE[i, j], Ymatrix[i, j]), T.arange(N), T.arange(td))

# update routine
updated_Ymatrix = Ymatrix - mf * gradE / divgradE
updatefcn = theano.funcion([Xmatrix, Ymatrix, mf], updated_Ymatrix)