from itertools import product

import numpy as np

# gradient
def tensor_gradient_element(fcn, X, indices, dx=None, tol=1e-8):
    if dx == None:
        tdx = min(0.1, abs(0.1 * X[indices])) if X[indices] != 0. else 0.1
        gradf_old = tensor_gradient_element(fcn, X, indices, dx=tdx)
        gradf_new = tensor_gradient_element(fcn, X, indices, dx=tdx * 0.1)
        while abs(gradf_new - gradf_old) > tol:
            gradf_old = gradf_new
            tdx *= 0.1
            gradf_new = tensor_gradient_element(fcn, X, indices, dx=tdx)
        return gradf_new
    else:
        dX = np.zeros(X.shape)
        dX[indices] = dx
        return (fcn(X + dX) - fcn(X - dX)) / (2 * dx)

def tensor_gradient(fcn, X, *args, **kwargs):
    gradfcnval = np.zeros(X.shape)
    for indices in product(*map(range, X.shape)):
        gradfcnval[indices] = tensor_gradient_element(fcn, X, indices, *args, **kwargs)
    return gradfcnval

# divergence of gradient (second derivatives)
def tensor_divgrad_element(fcn, X, indices, dx=None, tol=1e-12):
    if dx == None:
        tdx = min(0.1, abs(0.1 * X[indices])) if X[indices] != 0. else 0.1
        divgradf_old = tensor_divgrad_element(fcn, X, indices, dx=tdx)
        divgradf_new = tensor_divgrad_element(fcn, X, indices, dx=tdx * 0.1)
        while abs(divgradf_new - divgradf_old) > tol:
            divgradf_old = divgradf_new
            tdx *= 0.1
            divgradf_new = tensor_divgrad_element(fcn, X, indices, dx=tdx)
        return divgradf_new
    else:
        dX = np.zeros(X.shape)
        dX[indices] = dx
        return (fcn(X + dX) - 2*fcn(X) + fcn(X - dX)) / (dx*dx)

def tensor_divgrad(fcn, X, *args, **kwargs):
    divgradfcnval = np.zeros(X.shape)
    for indices in product(*map(range, X.shape)):
        divgradfcnval[indices] = tensor_divgrad_element(fcn, X, indices, *args, **kwargs)
    return divgradfcnval