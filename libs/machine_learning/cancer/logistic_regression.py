"Logistic regression classifier for breas cancer"
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from libs.machine_learning.views import prepare_dataset
from libs.machine_learning.views.prepare_dataset import DataSet

Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]
Solver = Literal["linear", "lbfgs"]


@dataclass
class BreastCancer:
    "BreastCancer classifier"

    n_components: Optional[int] = None
    random_state: int = 1
    solver: Solver = "lbfgs"
    n_splits: int = 10

    def _get_dataset(self) -> prepare_dataset.Data:
        "Get dataset for breast cancer"
        dataset = DataSet(name="cancer")
        return dataset.get_dataset()

    def _split(self, test_size: float = 0.2) -> prepare_dataset.SplittedData:
        "Split data"
        dataset = self._get_dataset()

        return dataset.split(
            test_size=test_size,
            stratify=False,
            random_state=self.random_state,
        )

    def train(self) -> None:
        "train model"
        x_train, x_test, y_train, y_test = self._split()
        param_range = [0.01, 0.01, 0.1, 1.0, 10.0, 100.0]
        pipe_lr = make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components),
            LogisticRegression(
                random_state=self.random_state,
                solver=self.solver,
            ),
        )
        train_scores, test_scores = validation_curve(
            estimator=pipe_lr,
            X=x_train,
            y=y_train,
            param_name="logisticregression__C",
            param_range=param_range,
            cv=10,
        )
        self.plot(
            x_axis=param_range,
            y_axis=train_scores,
            x_label="Accuracy training",
            color="blue",
        )
        self.plot(
            x_axis=param_range,
            y_axis=test_scores,
            x_label="Accuracy validation",
            color="green",
        )
        self.show()
        pipe_lr.fit(x_train, y_train)
        score = pipe_lr.score(x_test, y_test)
        print(f"Test Accuracy: {score}")

    def plot(
        self,
        x_axis: List[float],
        y_axis: Tensor,
        x_label: str,
        color: str,
    ) -> None:
        "plot"
        mean = np.mean(y_axis, axis=1)
        std = np.std(y_axis, axis=1)
        if "validation" in x_label:
            linestyle = "--"
        else:
            linestyle = "-"
        plt.plot(
            x_axis,
            mean,
            linestyle=linestyle,
            color=color,
            marker="o",
            markersize=5,
            label=x_label,
        )
        plt.fill_between(
            x_axis,
            mean + std,
            mean - std,
            alpha=0.15,
            color=color,
        )

    def show(self) -> None:
        "Show the plot"
        plt.grid()
        plt.xscale("log")
        plt.legend(loc="lower right")
        plt.xlabel("Param C")
        plt.ylabel("Accuracy")
        plt.ylim([0.8, 1.03])
        plt.show()
