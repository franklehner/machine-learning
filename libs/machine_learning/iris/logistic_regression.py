"""Logistic Regression classifier for iris dataset
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from libs.machine_learning.models import classifier
from libs.machine_learning.views import iris, plot, prepare_dataset

Tensor = np.ndarray
DataSet = Tuple[Tensor, Tensor, Tensor, Tensor]
Labels = Union[List[int], List[iris.Name]]
Solver = Literal[
    "lbfgs",
    "liblinear",
    "newton-cg",
    "newton-cholesky",
    "sag",
    "saga",
]
MultiClass = Literal[
    "auto",
    "ovr",
    "multinomial",
]


@dataclass
class FeaturesIndex:
    "Get feature indexes"
    labels: Labels
    indexes: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.get_indexes()

    def get_indexes(self) -> None:
        "Get the indexes of the labels"
        dataset = iris.Iris()
        if self.labels:
            for label in self.labels:
                if isinstance(label, int):
                    self.indexes.append(label)
                else:
                    if label in ("petal", "sepal"):
                        idx = dataset.name2idx(name=label)
                        if isinstance(idx, int):
                            self.indexes.append(idx)
                        elif isinstance(idx, list):
                            self.indexes.extend(idx)


@dataclass
class IrisClassifier:
    "Logistic Regression classifier for iris data"

    samples: Optional[int] = None
    features: Optional[FeaturesIndex] = None
    standardize: bool = True

    def get_iris_dataset(self) -> DataSet:
        "Get the iris dataset"
        if self.standardize:
            x_train, x_test, y_train, y_test = prepare_dataset.get_standardized_data(
                count=self.samples,
            )
        else:
            x_train, x_test, y_train, y_test = prepare_dataset.get_splitted_dataset(
                count=self.samples,
            )

        if self.features is not None:
            if len(self.features.indexes) > 1:
                x_train = x_train[:, self.features.indexes]
                x_test = x_test[:, self.features.indexes]
            else:
                x_train = x_train[:, self.features.indexes[0]]
                x_test = x_test[:, self.features.indexes[0]]

        return x_train, x_test, y_train, y_test

    def init_logistic_regression(
        self,
        regularize: float = 100.0,
        solver: Solver = "lbfgs",
        multi_class: MultiClass = "ovr",
        random_state: int = 1,
    ) -> classifier.LRC:
        "initalize Model"
        lrc = classifier.LRC(
            regularize=regularize,
            solver=solver,
            random_state=random_state,
            multi_class=multi_class,
        )

        return lrc

    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        lrc: classifier.LRC,
    ) -> classifier.LRC:
        "train the Logistic Regression"
        return lrc.fit(x_train=x_train, y_train=y_train)

    def predict(self, data: Tensor, lrc: classifier.LRC) -> Tensor:
        "Predict classes"
        return lrc.predict(data=data)

    def predict_proba(self, data: Tensor, lrc: classifier.LRC) -> Tensor:
        "predict probabilities of the data"
        return lrc.predict_proba(data=data)

    def plot(
        self,
        lrc: classifier.LRC,
        data: Tensor,
        target: Tensor,
        names: List[str],
    ) -> None:
        "plot decision"
        dataset = prepare_dataset.get_origin_iris_dataset()
        test_idx = range(105, 150)
        plot.plot_decision_regions(
            data=data,
            target=target,
            classifier=lrc,
            test_idx=test_idx,
            names=names,
        )
        if self.features is not None:
            names = dataset.idx2name(idx=self.features.indexes)
            plt.xlabel(names[0])
            plt.ylabel(names[1])
            plt.legend(loc="upper left")
            plt.show()

    @property
    def target_names(self) -> Tensor:
        "Get iris target names"
        dataset = prepare_dataset.get_origin_iris_dataset()

        return dataset.target_names
