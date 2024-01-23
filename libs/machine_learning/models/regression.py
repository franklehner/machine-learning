"""Model for linear regression
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np


Tensor = np.ndarray


@dataclass
class LinearRegressionGD:
    "Linear regression model"

    eta: float = 0.001
    n_iter: int = 20
    w_: Tensor = field(init=False)
    cost_: List[float] = field(init=False, default_factory=list)

    def fit(self, X: Tensor, y: Tensor) -> "LinearRegressionGD":  # pylint: disable=invalid-name
        "train data"
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            output = self.net_input(X=X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X: Tensor) -> Tensor:  # pylint: disable=invalid-name
        "netto input"
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: Tensor) -> Tensor:  # pylint: disable=invalid-name
        "predict value"
        return self.net_input(X=X)
