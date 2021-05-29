import numpy as np
from function_object import Function, MatrixFunction, MatrixGrad, MatrixConstant
from math import sin, cos, sinh, cosh

np.random.seed(42)

UNI_BI_VARIATE = [
    Function(
        f = lambda x: x[0]**2 + x[1]**4,
        gradient = lambda x: np.array([2.*x[0], 2.*x[1]]),
        hessian = lambda x: np.array([[2., 0.], [0., 2.]]),
        minimizers= [np.array([0., 0.])]
    ),
    Function(
        f = lambda x: x**4,
        gradient = lambda x: 4*x**3,
        hessian = lambda x: 12*x**2,
        minimizers= [0]
    ),
    Function(
        f = lambda x: x**4 - x**3,
        gradient = lambda x: 4*x**3 - 3*x**2,
        hessian = lambda x: 12*x**2 - 6*x,
        minimizers= [0.75]
    ),
    Function(
        f = lambda x: sin(x) + x**4,
        gradient = lambda x: cos(x) + 4*x**3,
        hessian = lambda x: -sin(x) + 12*x**2,
        minimizers= [-0.59198478817]
    ),
    Function(
        f = lambda x: cosh(x) + x**3,
        gradient = lambda x: sinh(x) + 3*x**2,
        hessian = lambda x: cosh(x) + 6*x,
        minimizers= [0, -5.01777749355]
    ),
]

GENERATING_MATRICES = [
   np.random.uniform(10, 100, size=(n, n))*10**9 for n in range(10, 20, 2)
]

POS_DEF_MATRICES = [X.T @ X for X in GENERATING_MATRICES]

VECTORS = [np.random.uniform(-2, 2, size=len(matrix)) for matrix in POS_DEF_MATRICES]

BEES = [A @ x for A, x in zip(POS_DEF_MATRICES, VECTORS)]

MULTIVARIATE = [
    Function(
        f = MatrixFunction(A, b),
        gradient = MatrixGrad(A, b),
        hessian = MatrixConstant(A),
        minimizers = [x_bar]
    )
    for A, x_bar, b in zip(POS_DEF_MATRICES, VECTORS, BEES)
]


