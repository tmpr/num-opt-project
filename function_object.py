from typing import Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class Function:
    f: Callable
    gradient: Callable
    hessian: Callable
    minimizer: np.array

    def __call__(self, x: np.array):
        return self.f(x)

@dataclass
class MatrixGrad:
    A: np.array
    b: np.array

    def __call__(self, x: np.array):
        return self.A @ x - self.b

@dataclass
class MatrixFunction:
    A: np.array
    b: np.array

    def __call__(self, x: np.array):
        return 1/2 * x.T @ self.A @ x - self.b@x

@dataclass
class MatrixConstant:
    A: np.array

    def __call__(self, x: np.array):
        return self.A

