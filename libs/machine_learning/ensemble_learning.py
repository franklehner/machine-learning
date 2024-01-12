"Ensemble learning usecase"
from dataclasses import dataclass, field
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from libs.machine_learning.models.ensemble import MajorityVotesClassifier
from libs.machine_learning.views import prepare_dataset

Tensor = np.ndarray
Vote = Literal["classlabel", "probabilty"]
Classifier = Union[Pipeline, DecisionTreeClassifier, MajorityVotesClassifier]


@dataclass
class EnsembleLearning:
    "Ensemble learning"

    vote: Vote
    test_size: float
    stratify: bool = False
    class_labels: List[str] = field(init=False, default_factory=list)

    def build_classifiers(self) -> List[Classifier]:
        "build different classifiers"
        clf1 = LogisticRegression(
            penalty="l2",
            C=0.001,
            solver="lbfgs",
            random_state=1,
            multi_class="multinomial",
        )
        clf2 = DecisionTreeClassifier(
            max_depth=1,
            criterion="entropy",
            random_state=0,
        )
        clf3 = KNeighborsClassifier(
            n_neighbors=1,
            p=2,
            metric="minkowski",
        )
        pipe1 = Pipeline(
            [
                ["sc", StandardScaler()],
                ["clf", clf1],
            ],
        )
        pipe3 = Pipeline(
            [
                ["sc", StandardScaler()],
                ["clf", clf3],
            ],
        )
        mv_clf = MajorityVotesClassifier(
            classifiers=[pipe1, clf2, pipe3],
            vote=self.vote,
        )
        self.class_labels = [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Majority Vote",
        ]
        return [pipe1, clf2, pipe3, mv_clf]

    def get_data(self) -> prepare_dataset.Data:
        "Get dataset"
        dataset = prepare_dataset.DataSet(name="cancer")

        return dataset.get_dataset()

    def prepare_data(
        self, dataset: prepare_dataset.Data
    ) -> prepare_dataset.SplittedData:
        "prepare data"
        x_train, x_test, y_train, y_test = dataset.split(
            test_size=self.test_size,
            stratify=self.stratify,
        )
        return x_train, x_test, y_train, y_test

    def train(self) -> None:
        "train data"
        classifiers = self.build_classifiers()
        dataset = self.get_data()
        x_train, x_test, y_train, y_test = self.prepare_data(
            dataset=dataset,
        )
        for clf, label in zip(classifiers, self.class_labels):
            scores = cross_val_score(
                estimator=clf,
                X=x_train,
                y=y_train,
                cv=10,
                scoring="roc_auc",
            )
            print(
                f"Classification rate: {scores.mean()} (+/- {scores.std()} [{label}])",
            )
        self.plot(
            clfs=classifiers,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

    def plot(
        self,
        clfs: List[Classifier],
        x_train: Tensor,
        x_test: Tensor,
        y_train: Tensor,
        y_test: Tensor,
    ) -> None:
        "plot roc curve"
        colors = [
            "black",
            "orange",
            "blue",
            "green",
        ]
        linestyle = [":", "--", "-.", "-"]
        for clf, label, clr, ls in zip(clfs, self.class_labels, colors, linestyle):
            y_pred = clf.fit(x_train, y_train).predict(x_test)
            fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
            roc_auc = auc(x=fpr, y=tpr)
            plt.plot(
                fpr, tpr, color=clr, linestyle=ls, label=f"{label} (auc={roc_auc})",
            )

        plt.legend(loc="lower right")
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=2,
        )
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid(alpha=0.5)
        plt.xlabel("False-Positive-Rate")
        plt.ylabel("True-Positive-Rate")
        plt.show()
