"""
File containing all 4 of the minimizers and the `Minimizer` base-class.
"""

from abc import abstractmethod

import numpy as np

from function_object import Function


class Minimizer:
    tolerance = 1e-8
    max_iter = 10_000
    step_search_tolerance = 100
    step_decay = 0.8
    c = 0.4

    def __init__(self, function: Function, x_0: np.array):
        self.f = function
        self.history = {
            "Gradient Norm": [],
        }
        self.x = x_0

    def minimize(self):
        """Minimize towards one of the (local) minimizers.

        Raises:
            TimeoutError: If the tolerance is not achieved within the class-defined number of maximum iterations.
        """
        for i in range(self.__class__.max_iter):

            direction = self.direction
            step_length = self.step_length(direction)

            self.history["Gradient Norm"].append(self.gradient_norm)

            if self.has_converged():
                return

            self.step(step_length, direction)

        raise TimeoutError(
            f"Minimizer did not converge within {self.__class__.max_iter} steps."
            f"Distance: {self.distance}, Gradient Norm: {self.gradient_norm}")

    def step(self, step_length: float, direction: np.array) -> None:
        self.x += step_length * direction

    @property
    def gradient_norm(self):
        return np.linalg.norm(self.f.gradient(self.x))

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
        c, step_decay = self.__class__.c, self.__class__.step_decay
        for _ in range(self.__class__.step_search_tolerance):

            left = self.f(x + alpha*p)
            right = self.f(x) + c * alpha * np.dot(self.f.gradient(x), p)

            if left <= right:
                return alpha
            else:
                alpha *= step_decay

        return alpha

    def has_converged(self) -> bool:
        """Check if optimizer has converged.

        Returns:
            bool: Whether or not gradient at current x is close to zero.
        """
        return self.gradient_norm <= self.__class__.tolerance


class Newton(Minimizer):

    @property
    def direction(self):
        return np.inner(self.inv_H, self.f.gradient(self.x))

    @property
    def inv_H(self):
        return -1/self.f.hessian(self.x) if self.f.is_univariate else -np.linalg.inv(self.f.hessian(self.x))


class QuasiNewton(Newton):

    def __init__(self, function: Function, x_0: np.array):
        super().__init__(function, x_0)

        if self.f.is_univariate:
            self.H = -1/self.f.hessian(self.x)
        else:
            self.H = np.linalg.inv(self.f.hessian(self.x))

    @property
    def inv_H(self):
        return -self.H

    def step(self, step_length: float, direction: np.array) -> None:
        new_x = self.x + step_length*direction

        s = new_x - self.x
        y = self.f.gradient(new_x) - self.f.gradient(self.x)
        self.update_H(s, y)

        self.x = new_x

    def update_H(self, s: np.array, y: np.array) -> None:
        """Update H according to the SR1 method."""

        difference = s - np.inner(self.H, y)
        numerator = np.outer(difference, difference)
        denominator = np.inner(difference, y)

        self.H = self.H + numerator/denominator


class ConjugateGradient(Minimizer):
    step_search_tolerance = 100_000
    tolerance = 1e-5
    min_step = float('inf')

    def __init__(self, function: Function, x_0: np.array):
        super().__init__(function, x_0)
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


class SteepestDescent(Minimizer):
    tolerance = 0.0001
    max_iter = 100_000

    @property
    def direction(self) -> np.array:
        return -self.f.gradient(self.x)
