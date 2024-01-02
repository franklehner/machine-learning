"""Use case for classifying the iris dataset via decision tree
classifier
"""
from dataclasses import dataclass, field
from typing import List, Literal, Tuple
import numpy as np

from libs.machine_learning.models.classifier import TreeClassifier
from libs.machine_learning.views import prepare_dataset
from libs.machine_learning.views.plot import plot_graph


Tensor = np.ndarray
Criterion = Literal["gini", "entropy"]
DataSet = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class IrisClassifier:
    "Decision Tree classifier for iris dataset"

    criterion: Criterion
    max_depth: int
    filename: str
    random_state: int = 1
    feature_names: List[str] = field(init=False, default_factory=list)
    class_names: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.feature_names, self.class_names = self.__get_names()

    def init_decision_tree(self) -> TreeClassifier:
        "Init the tree classifier"
        return TreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

    def get_iris_data(self) -> DataSet:
        "Get splitted iris data"
        return prepare_dataset.get_splitted_dataset()

    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        tree: TreeClassifier,
    ) -> TreeClassifier:
        "Train the model"
        return tree.fit(x_train=x_train, y_train=y_train)

    def predict(self, data: Tensor, tree: TreeClassifier) -> Tensor:
        "predict classes"
        return tree.predict(data)

    def plot_graph(
        self,
        tree: TreeClassifier,
        class_names: List[str],
        feature_names: List[str],
    ) -> None:
        "plot graph"
        plot_graph(
            dtree=tree,
            class_names=class_names,
            feature_names=feature_names,
            filename=self.filename,
        )

    def __get_names(self) -> Tuple[List[str], List[str]]:
        "get feature names and class names"
        iris = prepare_dataset.get_origin_iris_dataset()

        return list(iris.feature_names), list(iris.target_names)
