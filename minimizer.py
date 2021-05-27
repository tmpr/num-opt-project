"""
File containing all 4 of the minimizers and the `Minimizer` base-class.
"""

import numpy as np
from function_object import Function
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

REPORTS = Path('reports')

class Minimizer:
    tolerance = 1e-5
    max_iter = 1000

    def __init__(self, function: Function, x_0: np.array, c=0.4, step_decay=0.8):
        self.f = function
        self.history = []
        self.alphas = []
        self.c = c
        self.step_decay = step_decay
        self.x = x_0

    def minimize(self):
        
        for i in range(Minimizer.max_iter):
            self.history.append(self.x)
            if self.has_converged():
                self.generate_analytics()
                return

            direction = self.direction
            step_length = self.step_length(direction)
            self.alphas.append(step_length)

            self.x += step_length * direction

        self.generate_analytics()
        raise TimeoutError(f"Minimizer did not converge within {Minimizer.max_iter} steps.")


    def generate_analytics(self) -> None:
        distances = np.array([self.distance(x) for x in self.history])
        plt.plot(distances)
        plt.savefig(REPORTS / (self.__class__.__name__ +  str(datetime.now()) + '.png'))

    def distance(self, x):
        return np.linalg.norm(self.x - self.f.minimizer)

    @property
    def direction(self): raise NotImplementedError("Cannot call from base Minimizer.")

    def step_length(self, p: np.array, alpha=1):
        x = self.x

        left = self.f(x + alpha*p)
        right = self.f(x) + self.c * alpha *  np.dot(self.f.gradient(x), p)

        return alpha if left <=  right else self.step_length(p, self.step_decay*alpha)

    def has_converged(self) -> bool:
        return np.linalg.norm(self.x - self.f.minimizer) <= Minimizer.tolerance


class NewtonMinimizer(Minimizer):

    @property
    def direction(self):
        return -np.linalg.inv(self.f.hessian(self.x)) @ self.f.gradient(self.x)


class QuasiNewtonMinimizer(Minimizer):

    @property
    def direction(self):
        pass


class ConjugateGradientMinimizer(Minimizer):

    @property
    def direction(self):
        pass


class SteepestDescendMinimizer(Minimizer):

    @property
    def direction(self):
        return -self.f.gradient(self.x)