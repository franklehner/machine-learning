"""Perceptron
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import seed


@dataclass
class Perceptron:
    """Perceptron-classifier
    """

    eta: float = 0.01
    n_iter: int = 50
    random_state: int = 1
    weights_: np.ndarray = field(init=False)
    errors_: List[int] = field(init=False, default_factory=list)

    def fit(self, train_data: np.ndarray, train_target: np.ndarray) -> "Perceptron":
        "Adjust training data"
        rgen = np.random.RandomState(seed=self.random_state)  # pylint: disable=no-member
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + train_data.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(train_data, train_target):
                update = self.eta * (target - self.predict(data=xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, data: np.ndarray) -> np.ndarray:
        "Netto input"
        return np.dot(data, self.weights_[1:]) + self.weights_[0]

    def predict(self, data: np.ndarray) -> np.ndarray:
        "Return predicted class label"
        return np.where(self.net_input(data=data) >= 0.0, 1, -1)


@dataclass
class AdalineGD:
    """Adaptive linear neurons
    """
    eta: float = 0.01
    n_iter: int = 50
    weights: np.ndarray = field(init=False)
    costs: List[float] = field(init=False, default_factory=list)

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


@dataclass
class AdalineSGD:
    """Adaline with stochastic gradient descent
    """
    eta: float = 0.01
    n_iter: int = 10
    weights_initialized: bool = False
    shuffle: bool = True
    random_state: Optional[int] = None
    weights: np.ndarray = field(init=False)
    costs_: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.random_state:
            seed(self.random_state)

    def fit(self, data: np.ndarray, targets: np.ndarray) -> "AdalineSGD":
        """Fit train data
        """
        self._initial_weights(
            data=data.shape[1],
        )

        for _ in range(self.n_iter):
            if self.shuffle:
                data, targets = self._shuffle(
                    data=data,
                    targets=targets,
                )
            cost = []

            for date, target in zip(data, targets):
                cost.append(self._update_weights(data=date, targets=target))

            avg_cost = sum(cost) / len(targets)
            self.costs_.append(avg_cost)

        return self

    def partial_fit(self, data: np.ndarray, targets: np.ndarray) -> "AdalineSGD":
        """Fit train data without reinitializing the weights
        """
        if not self.weights_initialized:
            self._initial_weights(data.shape[1])
        if targets.ravel().shape[0] > 1:
            for element, target in zip(data, targets):
                self._update_weights(
                    data=element,
                    targets=target,
                )
        else:
            self._update_weights(data=data, targets=targets)

        return self

    def _initial_weights(self, data: int):
        """Fill weights with zero
        """
        self.weights = np.zeros(1 + data)
        self.weights_initialized = True

    def _shuffle(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """shuffle the training data and targets
        """
        permut = np.random.permutation(len(targets))

        return data[permut], targets[permut]

    def _update_weights(self, data: np.ndarray, targets: np.ndarray) -> float:
        """update the weights
        """
        output = self.net_input(data=data)
        error = targets - output
        self.weights[1:] += self.eta * data.dot(error)
        self.weights[0] += self.eta * error
        cost = 0.5 * error**2

        return cost

    def net_input(self, data: np.ndarray) -> np.ndarray:
        """Netto input into the net
        """
        return np.dot(data, self.weights[1:]) + self.weights[0]

    def activation(self, data: np.ndarray) -> np.ndarray:
        """Returns the net input
        """
        return self.net_input(data=data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """predict the class
        """
        return np.where(self.activation(data=data) >= 0.0, 1, -1)


class LogisticRegression:
    """Classifier with logistic LogisticRegression
    and gradient descent.
    """

    def __init__(self, eta=0.5, n_iter=100, random_state=1):
        "Constructor"
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Adjust training data
        """
        rgen = np.random.RandomState(seed=self.random_state)  # pylint: disable=no-member
        self.w_ = rgen.normal(
            loc=0.0, scale=0.01, size=1 + X.shape[1],
        )
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] = self.eta * X.T.dot(errors)
            self.w_[0] = self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        "Net input"
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        "Calculate logistic activation"
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        "Return class label"
        return np.where(self.net_input(X) >= 0.0, 1, 0)
