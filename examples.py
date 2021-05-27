import numpy as np
from function_object import Function, MatrixFunction, MatrixGrad, MatrixConstant

np.random.seed(42)

UNIVARIATE = [
    Function(
        f = lambda x: x[0]**2 + x[1]**4,
        gradient = lambda x: np.array([2.*x[0], 2.*x[1]]),
        hessian = lambda x: np.array([[2., 0.], [0., 2.]]),
        minimizer= np.array([0., 0.])
    )
]

GENERATING_MATRICES = [
   np.random.uniform(-2, 2, size=(n, n)) for n in range(2, 5, 1)
]

POS_DEF_MATRICES = [X.T @ X for X in GENERATING_MATRICES]

VECTORS = [np.random.uniform(-2, 2, size=len(matrix)) for matrix in POS_DEF_MATRICES]

BEES = [A @ x for A, x in zip(POS_DEF_MATRICES, VECTORS)]

MULTIVARIATE = [
    Function(
        f = MatrixFunction(A, b),
        gradient = MatrixGrad(A, b),
        hessian = MatrixConstant(A),
        minimizer = x_bar
    )
    for A, x_bar, b in zip(POS_DEF_MATRICES, VECTORS, BEES)
]


