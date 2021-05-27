"""
File containing all 4 of the minimizers and the `Minimizer` base-class.
"""

from dataclasses import dataclass
import numpy as np
from numpy.lib.utils import who
from function_object import Function
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import plotly.express as px
import pandas as pd


SUCCESS = Path('reports/success')
FAIL = Path('reports/fail')

class Minimizer:
    tolerance = 1e-2
    max_iter = 10_000
    step_search_tolerance = 50

    def __init__(self, function: Function, x_0: np.array, c=0.4, step_decay=0.8):
        self.f = function
        self.history = {
            "L2 Distance to Min" : [],
            "Step Length": [],
            "Direction Magnitude": []
        }
        self.c = c
        self.step_decay = step_decay
        self.x = x_0

    def minimize(self):
        
        for i in range(Minimizer.max_iter):
            

            direction = self.direction
            step_length = self.step_length(direction)
            
            self.history["L2 Distance to Min"].append(float(self.distance(self.x)))
            self.history["Step Length"].append(step_length)
            self.history["Direction Magnitude"].append(np.linalg.norm(direction))

            if self.has_converged():
                self.generate_analytics()
                return

            self.step(step_length, direction)

        self.generate_analytics()
        raise TimeoutError(f"Minimizer did not converge within {Minimizer.max_iter} steps. Distance: {self.distance(self.x)}")


    def step(self, step_length: float, direction: np.array) -> None:
        self.x += step_length * direction

    def generate_analytics(self) -> None:
        fig = px.line(data_frame=pd.DataFrame.from_dict(self.history), 
                      title="Optimization History of "+self.__class__.__name__)
        
        directory = SUCCESS if self.has_converged() else FAIL

        fig.write_html(str(directory / (self.__class__.__name__ +  str(datetime.now()) + '.html')))

    def distance(self, x):
        return np.linalg.norm(self.x - self.f.minimizer)

    @property
    def direction(self): 
        raise NotImplementedError("Cannot call from base Minimizer.")

    def step_length(self, p: np.array, alpha=1):
        x = self.x
        for i in range(__class__.step_search_tolerance):
        
            left = self.f(x + alpha*p)
            right = self.f(x) + self.c * alpha *  np.dot(self.f.gradient(x), p)

            if left <= right:
                return alpha
            else:
                alpha *= self.step_decay

        return alpha

    def has_converged(self) -> bool:
        return np.linalg.norm(self.x - self.f.minimizer) <= __class__.tolerance


class NewtonMinimizer(Minimizer):

    @property
    def direction(self):
        return -np.linalg.inv(self.f.hessian(self.x)) @ self.f.gradient(self.x)


class QuasiNewtonMinimizer(Minimizer):

    def __init__(self, function: Function, x_0: np.array, c=0.9, step_decay=0.95):
        super().__init__(function, x_0, c=c, step_decay=step_decay)
        # We only compute the Hessian explicitly once.
        self.H = -np.linalg.inv(self.f.hessian(self.x))

    @property
    def direction(self):
        return -self.H @ self.f.gradient(self.x)

    def step(self, step_length: float, direction: np.array) -> None:
        new_x = self.x + step_length*direction

        s = new_x - self.x
        y = self.f.gradient(new_x) - self.f.gradient(self.x)

        if not self.has_converged():
            self.update_H(s, y)

        self.x = new_x
        
    def update_H(self, s: np.array, y: np.array):
        """Update H according to the SR1 method."""
        difference = s - self.H@y
        numerator = np.outer(difference, difference)
        denominator = np.inner(difference, y)

        self.H = self.H + numerator/denominator



class ConjugateGradientMinimizer(Minimizer):

    @property
    def direction(self):
        pass

class SteepestDescentMinimizer(Minimizer):

    @property
    def direction(self):
        return -self.f.gradient(self.x)