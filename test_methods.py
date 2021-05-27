import pytest
import numpy as np

from examples import MULTIVARIATE, UNIVARIATE
from minimizer import *


@pytest.fixture(params=[NewtonMinimizer, 
                        QuasiNewtonMinimizer, 
                        SteepestDescentMinimizer, 
                        # ConjugateGradientMinimizer
                        ])
def method(request):
    return request.param


@pytest.mark.parametrize('function', MULTIVARIATE + UNIVARIATE)
def test_methods(function: Function, method: Minimizer):
    x_0 = np.ones(shape=function.minimizer.shape) * 100
    minimizer = method(function=function, x_0=x_0)
    minimizer.minimize()
    assert minimizer.has_converged()
