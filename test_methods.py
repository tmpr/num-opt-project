import pytest
import numpy as np

from examples import MULTIVARIATE, BIVARIATE
from minimizer import *


@pytest.fixture(params=[NewtonMinimizer, 
                        QuasiNewtonMinimizer, 
                        SteepestDescentMinimizer, 
                        # ConjugateGradientMinimizer
                        ])
def method(request):
    return request.param


@pytest.mark.parametrize('function', MULTIVARIATE + BIVARIATE)
def test_methods(function: Function, method: Minimizer):
    x_0 = np.ones(shape=function.minimizers[0].shape) * 100
    minimizer = method(function=function, x_0=x_0)
    minimizer.minimize()
    assert minimizer.has_converged()
