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

If this does not work, try out `python -m pytest -vv`, if there are problems with the path.


## Remarks

The tests can be found in `test_methods.py`, which imports its examples from `examples.py`.
A test is deemed successful if a minimizer has converged, which in turn means if it found
an x at which the gradient is close to 0.

If you want to see a plot of the optimization progress, the test-script produces an interactive plot
you can open in a browser.
