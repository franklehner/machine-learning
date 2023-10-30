"""Perceptron
"""
from __future__ import annotations
import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class Perceptron:
    """Perceptron classifier
    """
    eta: float = 0.01
    n_iter: int = 10
    w_: np.ndarray = dataclasses.field(init=False)
    errors_: List[int] = dataclasses.field(init=False, default_factory=list)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "Perceptron":
        """Fit training data
        """
        self.w_ = np.zeros(1 + x_train.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for x_train_i, target in zip(x_train, y_train):
                update = self.eta * (target - self.predict(x_train_i))
                self.w_[1:] += update * x_train_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def predict(self, x_train: np.ndarray):
        """Predict class
        """
        return np.where(self.net_input(x_train=x_train) >= 0.0, 1, -1)

    def net_input(self, x_train: np.ndarray):
        """Calculate netto input
        """
        return np.dot(x_train, self.w_[1:]) + self.w_[0]


@dataclasses.dataclass
class AdalineGD:
    """Adaptive linear neurons
    """
    eta: float = 0.01
    n_iter: int = 50
    weights: np.ndarray = dataclasses.field(init=False)
    costs: List[float] = dataclasses.field(init=False, default_factory=list)

    def fit(self, data: np.ndarray, targets: np.ndarray) -> "AdalineGD":
        """Training of data
        """
        self.weights = np.zeros(1 + data.shape[1])

        for _ in range(self.n_iter):
            output = self.net_input(data=data)
            errors = targets - output
            self.weights[1:] += self.eta * data.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            costs = (errors**2).sum() / 2.0
            self.costs.append(costs)

        return self

    def net_input(self, data: np.ndarray) -> np.ndarray:
        """Netto input
        """
        return np.dot(data, self.weights[1:]) + self.weights[0]

    def activation(self, data: np.ndarray) -> np.ndarray:
        """Activation function
        """
        return self.net_input(data=data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict
        """
        return np.where(self.activation(data) >= 0.0, 1, -1)
