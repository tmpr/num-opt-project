from dataclasses import dataclass
from numbers import Number
from typing import Callable, List

import numpy as np


@dataclass
class Function:
    f: Callable
    gradient: Callable
    hessian: Callable
    minimizers: List[np.ndarray]

    def __call__(self, x: np.ndarray):
        return self.f(x)

    @property
    def is_univariate(self) -> bool:
        return isinstance(self.minimizers[0], Number)


@dataclass
class MatrixGrad:
    A: np.ndarray
    b: np.ndarray

    def __call__(self, x: np.ndarray):
        return (self.A @ x) - self.b


@dataclass
class MatrixFunction:
    A: np.ndarray
    b: np.ndarray

    def __call__(self, x: np.ndarray):
        return 1/2 * np.inner(x, self.A @ x) - np.inner(self.b, x)


@dataclass
class MatrixConstant:
    A: np.ndarray

    def __call__(self, x: np.ndarray):
        return self.A
