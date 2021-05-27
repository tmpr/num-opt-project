import pytest
import numpy as np

from examples import MULTIVARIATE, UNIVARIATE
from minimizer import *


@pytest.fixture(params=[NewtonMinimizer, 
                        # QuasiNewtonMinimizer, 
                        SteepestDescendMinimizer, 
                        # ConjugateGradientMinimizer
                        ])
def method(request):
    return request.param


@pytest.mark.parametrize('function', [*MULTIVARIATE])
def test_multivariate(function: Function, method: Minimizer):
    x_0 = np.ones_like(function.minimizer)
    minimizer = method(function, x_0)
    minimizer.minimize()
    assert minimizer.has_converged()

@pytest.mark.parametrize('function', [*UNIVARIATE])
def test_univariate(function: Function, method: Minimizer):
    x_0 = np.ones_like(function.minimizer)
    minimizer = method(function, x_0)
    minimizer.minimize()
    assert minimizer.has_converged()
