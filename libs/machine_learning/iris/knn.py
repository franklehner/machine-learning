"""K - Nearest Neighbors classfier
"""
from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np

from libs.machine_learning.models.classifier import KNN, KNeighborsClassifier
from libs.machine_learning.views import prepare_dataset

Tensor = np.ndarray
DataSet = Tuple[Tensor, Tensor, Tensor, Tensor]
POW = Literal[1, 2]


@dataclass
class IrisClassifier:
    "K - Nearest Neighbors classifier for iris data"

    n_neighbors: int = 5
    p_param: POW = 2
    knn: KNN = field(init=False)

    def __post_init__(self):
        self.knn = KNN(
            n_neighbors=self.n_neighbors,
            p=self.p_param,
            metric="minkowski",
        )

    def get_iris_data(self) -> DataSet:
        "get iris data"
        return prepare_dataset.get_splitted_dataset()

    def train(self, x_train: Tensor, y_train: Tensor) -> KNeighborsClassifier:
        "Train data"
        return self.knn.classifier.fit(X=x_train, y=y_train)

    def predict(self, data: Tensor) -> Tensor:
        "predict data"
        return self.knn.classifier.predict(X=data)
