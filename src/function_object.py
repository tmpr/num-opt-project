from sys import hexversion
from typing import Callable
import numpy as np
from dataclasses import dataclass

from numpy.lib.function_base import gradient

@dataclass
class Function:
    f: Callable
    gradient: Callable
    hessian: Callable
    minimizer: np.array

    def __call__(self, x: np.array):
        return self.f(x)