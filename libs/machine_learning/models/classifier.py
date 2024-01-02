"""Classes for different classifiers
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

Tensor = np.ndarray
POW = Literal[1, 2]


@dataclass
class LRC:
    "Logistic regression classifier"
    regularize: float
    solver: str
    random_state: int
    multi_class: str
    classifier: LogisticRegression = field(init=False)

    def __post_init__(self):
        self.classifier = LogisticRegression(
            C=self.regularize,
            solver=self.solver,
            random_state=self.random_state,
            multi_class=self.multi_class,
        )

    def fit(self, x_train: Tensor, y_train: Tensor) -> "LRC":
        "Train data"
        return self.classifier.fit(x_train, y_train)

    def predict(self, data: Tensor) -> Tensor:
        "predict data"
        return self.classifier.predict(X=data)

    def predict_proba(self, data: Tensor) -> Tensor:
        "predict probabilities of the classes"
        return self.classifier.predict_proba(X=data)

    def coefficients(self) -> Tensor:
        "Return coefficients"
        return self.classifier.coef_


@dataclass
class SupportVectorMachine:
    "Support Vector Machine classifier"

    regularize: float
    gamma: float
    random_state: int
    kernel: str
    classifier: SVC = field(init=False)

    def __post_init__(self):
        self.classifier = SVC(
            C=self.regularize,
            kernel=self.kernel,
            random_state=self.random_state,
            gamma=self.gamma,
        )

    def fit(self, x_train: Tensor, y_train: Tensor) -> "SupportVectorMachine":
        "train data"
        return self.classifier.fit(X=x_train, y=y_train)

    def predict(self, data: Tensor) -> Tensor:
        "predict classes of the data"
        return self.classifier.predict(X=data)


@dataclass
class TreeClassifier:
    "Decision Tree Classifier"

    criterion: str = "entropy"
    max_depth: int = 4
    random_state: int = 1
    classifier: DecisionTreeClassifier = field(init=False)

    def __post_init__(self):
        self.classifier = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

    def fit(self, x_train: Tensor, y_train: Tensor) -> "TreeClassifier":
        "Train the data"
        return self.classifier.fit(X=x_train, y=y_train)

    def predict(self, data: Tensor) -> Tensor:
        "predict class"
        return self.classifier.predict(X=data)


@dataclass
class RFC:
    "Random Forest Classifier"

    criterion: str = "entropy"
    n_estimators: int = 25
    random_state: int = 1
    n_jobs: int = 1
    classifier: RandomForestClassifier = field(init=False)

    def __post_init__(self):
        self.classifier = RandomForestClassifier(
            criterion=self.criterion,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, x_train: Tensor, y_train: Tensor) -> RandomForestClassifier:
        "train the data"
        return self.classifier.fit(X=x_train, y=y_train)


@dataclass
class KNN:
    "KNeighborsClassifier"

    n_neighbors: int = 5
    p: POW = 2
    metric: str = "minkowski"
    classifier: KNeighborsClassifier = field(init=False)

    def __post_init__(self):
        self.classifier = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            p=self.p,
        )

    def fit(self, x_train: Tensor, y_train: Tensor) -> KNeighborsClassifier:
        "Train data"
        return self.classifier.fit(X=x_train, y=y_train)
