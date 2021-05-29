# Numerical Optimization Project

4 Minimizers implemented to solve 10 optimization problems.

## Dependencies

Clearly, python. Further, install the following:

```
pip install pandas
pip install numpy
pip install plotly
pip install pytest
```

## Running the tests

Navigate to this directory and call `pytest`.


## Remarks

The tests can be found in `test_methods.py`, which imports its examples from `examples.py`.
A test is deemed successful if a minimizer has converged, which in turn means if it minimized
its L2-distance from the true minimum to less than `self.__class__.tolerance` within `self.__class__.max_iter`.

Optimizers have different tolerances and settings. For example, since they do converge but are slow, both `SteepestDescentMinimizer` and `ConjugateGradientMinimizer` have a `tolerance = 0.5`. Nevertheless, the optimization progress can be viewed in the generated graphs: Clearly we see that the minimizers approach the true minimizers.