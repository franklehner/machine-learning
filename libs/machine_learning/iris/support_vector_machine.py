"""Iris classifier with support vector machine
"""
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from libs.machine_learning.models.classifier import SupportVectorMachine
from libs.machine_learning.views import prepare_dataset
from libs.machine_learning.views.plot import plot_decision_regions

Tensor = np.ndarray
Label = Literal["petal", "sepal"]
Labels = Tuple[Label, ...]
Attribute = Literal["width", "length"]
Attributes = Tuple[Attribute, ...]
DataSet = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class FeaturesIndex:
    "Convert labels and dims to index"

    labels: Optional[Labels] = None
    attributes: Optional[Attributes] = None

    def __post_init__(self):
        if not self.labels:
            if self.attributes:
                raise ValueError("Attributes without label is not allowed")

    def get_indexes(self) -> List[int]:
        "Get the indexes of the labels"
        dataset = prepare_dataset.get_origin_iris_dataset()
        indexes = []
        if self.labels:
            for label in self.labels:
                if not self.attributes:
                    idx = dataset.name2idx(name=label)
                    if isinstance(idx, int):
                        indexes.append(idx)
                    elif isinstance(idx, list):
                        indexes.extend(idx)
                else:
                    for attribute in self.attributes:
                        idx = dataset.name2idx(name=label, dim=attribute)
                        if isinstance(idx, int):
                            indexes.append(idx)
                        elif isinstance(idx, list):
                            indexes.extend(idx)
        else:
            indexes = [idx for idx, _ in enumerate(dataset.feature_names)]

        return indexes

    @property
    def indexes(self) -> List[int]:
        "Get feature indexes"
        return self.get_indexes()


@dataclass
class IrisClassifier:
    "Support Vector Machine classifier for iris dataset"
    indexes: List[int]

    def get_iris_dataset(self) -> DataSet:
        "Get iris data splitted and standardized"
        x_train, x_test, y_train, y_test = prepare_dataset.get_standardized_data()
        x_train = x_train[:, self.indexes]
        x_test = x_test[:, self.indexes]

        return x_train, x_test, y_train, y_test

    def init_support_vector_machine(
        self,
        regularize: float,
        kernel: str,
        random_state: int,
        gamma: float,
    ) -> SupportVectorMachine:
        "init support vector machine"
        svm = SupportVectorMachine(
            regularize=regularize,
            gamma=gamma,
            random_state=random_state,
            kernel=kernel,
        )

        return svm

    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        svm: SupportVectorMachine,
    ) -> SupportVectorMachine:
        "Train the model"
        return svm.fit(x_train=x_train, y_train=y_train)

    def predict(self, data: Tensor, svm: SupportVectorMachine) -> Tensor:
        "Predict classes"
        return svm.predict(data=data)

    def plot(
        self,
        svm: SupportVectorMachine,
        data: Tensor,
        target: Tensor,
        names: List[str],
    ) -> None:
        "plot decision"
        dataset = prepare_dataset.get_origin_iris_dataset()
        test_idx = range(105, 150)
        plot_decision_regions(
            data=data,
            target=target,
            classifier=svm,
            test_idx=test_idx,
            names=names,
        )
        names = dataset.idx2name(idx=self.indexes)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    @property
    def target_names(self) -> Tensor:
        "Get iris target names"
        dataset = prepare_dataset.get_origin_iris_dataset()

        return dataset.target_names
