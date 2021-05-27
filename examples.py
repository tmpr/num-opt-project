import numpy as np
from function_object import Function, MatrixFunction, MatrixGrad, MatrixConstant

np.random.seed(42)

BIVARIATE = [
    Function(
        f = lambda x: x[0]**2 + x[1]**4,
        gradient = lambda x: np.array([2.*x[0], 2.*x[1]]),
        hessian = lambda x: np.array([[2., 0.], [0., 2.]]),
        minimizers= [np.array([0., 0.])]
    )
]

GENERATING_MATRICES = [
   np.random.uniform(-100, 100, size=(n, n)) for n in range(10, 20, 2)
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


