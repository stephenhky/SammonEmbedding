import numpy as np
import theano
import theano.tensor as T

# define variables
# number of points
N = T.iscalar('N')
# dimensions
d = T.iscalar('d')
# target dimensions
td = T.iscalar('td')

# dummy variables
i = T.arange(N)
j = T.arange(N)

# coordinate variables
Xmatrix = T.dmatrix('Xmatrix')
Ymatrix = T.dmatrix('Ymatrix')

# distance function
dist = T.sqrt(T.sqr(Xmatrix[i]-Xmatrix[j]))
tdist = T.sqrt(T.sqr(Ymatrix[i]-Ymatrix[j]))

# cost function
