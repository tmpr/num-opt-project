import pytest
import numpy as np

from examples import MULTIVARIATE, UNI_BI_VARIATE
from minimizer import *
from shutil import rmtree
from pathlib import Path


from numbers import Number


def test_delete_reports():
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports/'success').mkdir()
    (reports/'fail').mkdir()
    rmtree(reports/'success')
    rmtree(reports/'fail')
    (reports/'success').mkdir()
    (reports/'fail').mkdir()



@pytest.fixture(params=[Newton, 
                        QuasiNewton, 
                        SteepestDescent, 
                        ConjugateGradient
                        ])
def method(request):
    return request.param


@pytest.mark.parametrize('function', UNI_BI_VARIATE + MULTIVARIATE)
def test_methods(function: Function, method: Minimizer):
    x_0 = np.ones(shape=function.minimizers[0].shape) * 3 if not function.is_univariate else 5
    minimizer = method(function=function, x_0=x_0)
    minimizer.minimize()
    
    assert minimizer.has_converged()
