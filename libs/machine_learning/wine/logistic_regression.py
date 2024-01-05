"""Classify wine data with pca
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from libs.machine_learning.models.classifier import LogisticRegression
from libs.machine_learning.views import prepare_dataset
from libs.machine_learning.views.plot import plot_decision_regions

MultiClass = Literal["ovr", "multinomial"]
Solver = Literal["linear", "lbfgs"]
Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class WineClassifier:
    "Classifier for wine data"

    multiclass: MultiClass
    solver: Solver
    n_components: Optional[int] = None
    plot: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.n_components == 2:
            self.plot = True

    def get_data(self) -> TrainTest:
        "Get splitted and standardized wine data"
        return prepare_dataset.get_standardized_wine_data()

    def train(self) -> None:
        "Train Model and print and plot"
        x_train, x_test, y_train, y_test = self.get_data()
        pca = PCA(n_components=self.n_components)
        lr = LogisticRegression(
            multi_class=self.multiclass,
            random_state=1,
            solver=self.solver,
        )
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)
        lr.fit(X=x_train_pca, y=y_train)
        if self.plot:
            self.plot_decisions(
                data=x_train_pca,
                target=y_train,
                classifier=lr,
            )
            self.plot_decisions(
                data=x_test_pca,
                target=y_test,
                classifier=lr,
            )

        y_pred = lr.predict(x_test_pca)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Accuracy: {score}")
        print(f"PCA variances:\n{pca.explained_variance_}")

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
