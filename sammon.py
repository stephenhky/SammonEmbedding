import numpy as np
import theano
import theano.tensor as T

# define variables
# target dimensions
td = T.iscalar('td')

# coordinate variables
Xmatrix = T.dmatrix('Xmatrix')
Ymatrix = T.dmatrix('Ymatrix')

# number of points and dimensions
N, d = T.shape(Xmatrix)

# dummy variables
j = T.arange(N)
i = T.arange(j)

# distance function (Euclidean distance)
dist = T.sqrt(T.sqr(Xmatrix[i]-Xmatrix[j]))
tdist = T.sqrt(T.sqr(Ymatrix[i]-Ymatrix[j]))

# cost function
