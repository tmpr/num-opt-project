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
    (reports/'success').mkdir(exist_ok=True)
    (reports/'fail').mkdir(exist_ok=True)
    rmtree(reports/'success')
    rmtree(reports/'fail')
    (reports/'success').mkdir(exist_ok=True)
    (reports/'fail').mkdir(exist_ok=True)


@pytest.fixture(params=[Newton,
                        QuasiNewton,
                        SteepestDescent,
                        ConjugateGradient
                        ])
def method(request):
    return request.param


@pytest.mark.parametrize('function', UNI_BI_VARIATE + MULTIVARIATE)
def test_methods(function: Function, method: Minimizer):
    x_0 = np.ones(
        shape=function.minimizers[0].shape) * 100 if not function.is_univariate else 100
    minimizer = method(function=function, x_0=x_0)
    minimizer.minimize()

    assert minimizer.has_converged()
