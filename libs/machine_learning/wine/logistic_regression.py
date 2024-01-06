"""Classify wine data with pca
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from libs.machine_learning.models.classifier import LogisticRegression
from libs.machine_learning.views import prepare_dataset
from libs.machine_learning.views.plot import plot_decision_regions

MultiClass = Literal["ovr", "multinomial"]
Solver = Literal["linear", "lbfgs"]
Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]
Reduction = Literal["pca", "lda", "kernel-pca"]
Reducer = Union[KernelPCA, PCA, LinearDiscriminantAnalysis]
Kernel = Literal["rbf", "linear"]


@dataclass
class WineClassifier:
    "Classifier for wine data"

    multiclass: MultiClass
    solver: Solver
    reduction: Reduction
    n_components: Optional[int] = None
    kernel: Optional[Kernel] = None
    gamma: Optional[float] = None
    plot: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.reduction == "kernel-pca":
            if not all([self.kernel, self.gamma]):
                raise ValueError("kernel and gamma must be set")

        if self.reduction == "lda":
            data = prepare_dataset.get_origin_wine_dataset()
            n_classes = len(np.unique(data.targets))
            if self.n_components > n_classes -1:
                self.n_components = n_classes - 1

        if self.n_components == 2:
            self.plot = True

    def get_data(self) -> TrainTest:
        "Get splitted and standardized wine data"
        return prepare_dataset.get_standardized_wine_data()

    def reduce(self) -> Reducer:
        "choose reduction method"
        if self.reduction == "pca":
            return PCA(n_components=self.n_components)
        if self.reduction == "lda":
            return LinearDiscriminantAnalysis(n_components=self.n_components)

        return KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
        )

    def train(self) -> None:
        "Train Model and print and plot"
        x_train, x_test, y_train, y_test = self.get_data()
        reducer = self.reduce()
        lr = LogisticRegression(
            multi_class=self.multiclass,
            random_state=1,
            solver=self.solver,
        )
        x_train_reduced = reducer.fit_transform(x_train, y_train)
        x_test_reduced = reducer.transform(x_test)
        lr.fit(X=x_train_reduced, y=y_train)
        if self.plot:
            self.plot_decisions(
                data=x_train_reduced,
                target=y_train,
                classifier=lr,
            )
            self.plot_decisions(
                data=x_test_reduced,
                target=y_test,
                classifier=lr,
            )

        y_pred = lr.predict(x_test_reduced)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Accuracy: {score}")
        print(f"{self.reduction.upper()} variances:\n{reducer.explained_variance_ratio_}")

    def plot_decisions(self, data: Tensor, target: Tensor, classifier: LogisticRegression):
        "Plot decisions"
        plot_decision_regions(
            data=data,
            target=target,
            classifier=classifier,
        )
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()
