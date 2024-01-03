"""Sequential Backward Selection (SBS)
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

Estimator = Union[
    RandomForestClassifier,
    LogisticRegression,
    KNeighborsClassifier,
    SVC,
    DecisionTreeClassifier,
]
Tensor = np.ndarray
Score = Callable[[Tensor, Tensor], float]
Indices = Tuple[int, ...]


@dataclass
class SBS:
    "Sequential Backward Selection"

    estimator: Estimator
    k_features: int
    score: Score
    indices: Indices = field(init=False)
    subset: List[Indices] = field(init=False, default_factory=list)
    scores: List[float] = field(init=False, default_factory=list)
    k_scores: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.estimator = clone(self.estimator)

    def _calc_score(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        y_test: Tensor,
        indices: Indices,
    ) -> float:
        "Calc score"
        self.estimator.fit(
            x_train[:, indices], y_train,
        )
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.score(y_test, y_pred)

        return score

    def fit(
        self, data: Tensor, target: Tensor, test_size: float = 0.25, rs: int = 1
    ) -> "SBS":
        "train data"
        x_train, x_test, y_train, y_test = train_test_split(
            data,
            target,
            test_size=test_size,
            random_state=rs,
        )
        dim = x_train.shape[1]
        self.indices = tuple(range(dim))
        self.subset = [self.indices]
        score = self._calc_score(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            indices=self.indices,
        )
        self.scores = [score]

        while dim > self.k_features:
            scores: List[float] = []
            subsets: List[Indices] = []
            for perm in combinations(self.indices, r=dim - 1):
                score = self._calc_score(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    indices=perm,
                )
                scores.append(score)
                subsets.append(perm)

            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subset.append(self.indices)
            dim -= 1
            self.scores.append(scores[best])

        self.k_scores = [self.scores[-1]]

        return self
