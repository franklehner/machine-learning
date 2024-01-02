"Random Forest Classifier"
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

from libs.machine_learning.models.classifier import RFC, RandomForestClassifier
from libs.machine_learning.views import prepare_dataset

Tensor = np.ndarray
Criterion = Literal["gini", "entropy"]
DataSet = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class IrisClassifier:
    "Random Forest Classifier for iris dataset"

    criterion: Criterion
    n_estimators: Optional[int]
    n_jobs: int
    random_state: int = 1
    rfc: RFC = field(init=False)

    def __post_init__(self):
        if self.n_estimators is None:
            iris = prepare_dataset.get_origin_iris_dataset()
            n_estimators = math.ceil(math.sqrt(iris.data.shape[0])) * 2
        else:
            n_estimators = self.n_estimators

        self.rfc = RFC(
            criterion=self.criterion,
            n_estimators=n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def get_iris_data(self) -> DataSet:
        "Get iris data"
        return prepare_dataset.get_splitted_dataset()

    def train(self, x_train: Tensor, y_train: Tensor) -> RandomForestClassifier:
        "Train data"
        return self.rfc.classifier.fit(X=x_train, y=y_train)

    def predict(self, data: Tensor) -> Tensor:
        "predict data"
        return self.rfc.classifier.predict(X=data)
