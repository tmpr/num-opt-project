# Numerical Optimization Project

4 Minimizers implemented to solve 10 optimization problems.

## Dependencies

Clearly, you will need Python 🐍. Further, please install the following
by entering the following commands into your shell:

```
pip install pandas
pip install numpy
pip install plotly
pip install pytest
```

# Grading info

When I ran the testing script with the defined tolerances, all 10 examples passed the tests for all methods.

## Running the tests

Navigate to this directory and call `pytest -vv`. It will return a neat
summary of all the functions that passed.


## Remarks

The tests can be found in `test_methods.py`, which imports its examples from `examples.py`.
A test is deemed successful if a minimizer has converged, which in turn means if it minimized
its L2-distance from the true minimum to less than `self.__class__.tolerance` within `self.__class__.max_iter`. Each test starts from a vector where each coordinate is 1.
