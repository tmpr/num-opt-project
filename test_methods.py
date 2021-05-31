import pytest
import numpy as np

from examples import MULTIVARIATE, UNI_BI_VARIATE
from minimizer import *
from shutil import rmtree
from pathlib import Path
import plotly.express as px

REPORTS = Path('reports')
SUCCESS = REPORTS / 'success'
FAIL = REPORTS / 'fail'


def test_delete_reports():
    REPORTS.mkdir(exist_ok=True)
    rmtree(SUCCESS, ignore_errors=True)
    rmtree(REPORTS, ignore_errors=True)
    REPORTS.mkdir(exist_ok=True)
    SUCCESS.mkdir(exist_ok=True)


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
        shape=function.minimizers[0].shape) if not function.is_univariate else 1
    minimizer = method(function=function, x_0=x_0)
    minimizer.minimize()
    generate_analytics(minimizer)

    assert minimizer.has_converged()


def generate_analytics(minimizer: Minimizer) -> None:
    """Generate Plot to inspect optimization process."""

    fig = px.line(data_frame=pd.DataFrame.from_dict(minimizer.history),
                  title="Optimization History of "+minimizer.__class__.__name__)

    directory = SUCCESS if minimizer.has_converged() else FAIL

    fig.write_html(
        str(directory / (minimizer.__class__.__name__ + str(datetime.now()) + '.html')))
