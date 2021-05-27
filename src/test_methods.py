import pytest

from conjugate_gradient import conjugate_gradient
from line_search import line_search
from newton import newton
from quasi_newton import quasi_newton
from steepest_descent import steepest_descent


@pytest.fixture(params=[quasi_newton, newton, steepest_descent, conjugate_gradient])
def method(request):
    return request.param

class TestMultiVariateProblems:

    def test_something(self, method):
        assert method('dummy') is None

class TestUniBiVariateProblems:

    def test_something(self, method):
        assert method('dummy') is None


