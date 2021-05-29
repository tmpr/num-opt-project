"""
File containing all 4 of the minimizers and the `Minimizer` base-class.
"""

from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from function_object import Function

SUCCESS = Path('reports/success')
FAIL = Path('reports/fail')

class Minimizer:
    tolerance = 1e-7
    max_iter = 1000
    step_search_tolerance = 10
    min_step = 1e-3

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
        """Minimize towards one of the (local) minimizers.

        Raises:
            TimeoutError: If the tolerance is not achieved within the class-defined number of maximum iterations.
        """
        for i in range(self.__class__.max_iter):
            

            direction = self.direction
            step_length = self.step_length(direction)
            
            self.history["L2 Distance to Min"].append(float(self.distance))
            self.history["Step Length"].append(step_length)
            self.history["Direction Magnitude"].append(np.linalg.norm(direction))

            if self.has_converged():
                self.generate_analytics()
                return

            self.step(step_length, direction)

        self.generate_analytics()
        raise TimeoutError(f"Minimizer did not converge within {self.__class__.max_iter} steps. Distance: {self.distance}")


    def step(self, step_length: float, direction: np.array) -> None:
        self.x += step_length * direction

    def generate_analytics(self) -> None:
        """Generate Plot to inspect optimization process."""

        fig = px.line(data_frame=pd.DataFrame.from_dict(self.history), 
                      title="Optimization History of "+self.__class__.__name__)
        
        directory = SUCCESS if self.has_converged() else FAIL

        fig.write_html(str(directory / (self.__class__.__name__ +  str(datetime.now()) + '.html')))

    @property
    def distance(self) -> float:
        """Least distance of current x to one of the known minimizers."""
        return min(np.linalg.norm(self.x - minimizer) for minimizer in self.f.minimizers)

    @property
    @abstractmethod
    def direction(self) -> np.array: pass

    def step_length(self, p: np.array, alpha=1.) -> float:
        """Compute step length according to the book.

        Args:
            p (np.array): Direction
            alpha (int, optional): Initial step length. Defaults to 1.

        Returns:
            float: Step length
        """
        x = self.x
        for _ in range(self.__class__.step_search_tolerance):
        
            left = self.f(x + alpha*p)
            right = self.f(x) + self.c * alpha *  np.dot(self.f.gradient(x), p)

            if left <= right:
                return alpha
            else:
                alpha *= self.step_decay

        return max(alpha, self.__class__.min_step)

    def has_converged(self) -> bool:
        """Check if optimizer has converged.

        Returns:
            bool: Whether or not x is very close to minimizer.
        """
        return self.distance <= self.__class__.tolerance


class NewtonMinimizer(Minimizer):

    @property
    def direction(self):
        if not self.f.is_univariate:
            return -np.linalg.inv(self.f.hessian(self.x)) @ self.f.gradient(self.x)
        else:
            return -1/self.f.hessian(self.x) * self.f.gradient(self.x)


class QuasiNewtonMinimizer(Minimizer):

    def __init__(self, function: Function, x_0: np.array, c=0.9, step_decay=0.95):
        super().__init__(function, x_0, c=c, step_decay=step_decay)

        self.H = -1/self.f.hessian(self.x) if self.f.is_univariate else -np.linalg.inv(self.f.hessian(self.x))

    @property
    def direction(self) -> np.array:
        if not self.f.is_univariate:
            return -self.H @ self.f.gradient(self.x)
        else:
            return -self.H * self.f.gradient(self.x)

    def step(self, step_length: float, direction: np.array) -> None:
        new_x = self.x + step_length*direction

        s = new_x - self.x
        y = self.f.gradient(new_x) - self.f.gradient(self.x)

        if not self.has_converged():
            self.update_H(s, y)

        self.x = new_x
        
    def update_H(self, s: np.array, y: np.array) -> None:
        """Update H according to the SR1 method."""

        difference = s - (self.H*y if self.f.is_univariate else self.H@y)
        numerator = np.outer(difference, difference)
        denominator = np.inner(difference, y)

        self.H = self.H + numerator/denominator



class ConjugateGradientMinimizer(Minimizer):
    step_search_tolerance = 100_000
    max_iter = 10_000
    tolerance = 1
    min_step = float('inf')

    def __init__(self, function: Function, x_0: np.array, c=0.5, step_decay=0.99):
        super().__init__(function, x_0, c=c, step_decay=step_decay)
        self._direction = -self.f.gradient(x_0)

    def step(self, step_length: float, direction: np.array) -> None:
        direction = self._direction
        grad = self.f.gradient(self.x)
        
        self.x += step_length*direction
        new_grad = self.f.gradient(self.x)
        beta = np.inner(new_grad, new_grad - grad) / (np.linalg.norm(grad)**2)
        self._direction = -new_grad + beta * direction
        

    @property
    def direction(self) -> np.array:
        return self._direction


class SteepestDescentMinimizer(Minimizer):
    tolerance = 1
    step_search_tolerance = 1000
    max_iter = 10_000
    momentum = 1

    # We are veeery generous with Gradient Descent, as it is horribly slow.
    # However it is doing its job and continually gets closer to the 
    # minimum.

    def __init__(self, function: Function, x_0: np.array, c=0.9, step_decay=0.9):
        super().__init__(function, x_0, c=c, step_decay=step_decay)
        self.x_pre = None

    @property
    def direction(self) -> np.array:
        return -self.f.gradient(self.x)

    def step(self, step_length: float, direction: np.array) -> None:
        if self.x_pre is None:
            momentum_term = 0
        else:
            momentum_term = self.__class__.momentum*(self.x - self.x_pre)
        self.x_pre = self.x
        self.x += step_length*direction - momentum_term

