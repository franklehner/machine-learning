"SBS Classifier"
from dataclasses import dataclass
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from libs.machine_learning.models import sbs
from libs.machine_learning.views import prepare_dataset


Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]
ClassifierNames = Literal[
    "knn", "svm", "tree", "logres", "random_forest",
]
CLASSIFIERS = {
    "knn": sbs.KNeighborsClassifier,
    "svm": sbs.SVC,
    "tree": sbs.DecisionTreeClassifier,
    "logres": sbs.LogisticRegression,
    "random_forest": sbs.RandomForestClassifier,
}


@dataclass
class WineClassifier:
    "Wine classifier"

    classifier_name: ClassifierNames
    k_features: int

    def get_wine_data(self) -> prepare_dataset.Wine:
        "Get the wine data"
        return prepare_dataset.get_origin_wine_dataset()

    def _get_estimator(self) -> sbs.Estimator:
        "Get the estimator"
        model = CLASSIFIERS[self.classifier_name]

        return model()

    def train_on_best_features(
        self, estimator: sbs.Estimator, x_train: Tensor, y_train: Tensor
    ) -> sbs.Estimator:
        "Train estimator on best features"
        return estimator.fit(
            X=x_train,
            y=y_train,
        )

    def get_trained_sbs(
        self,
        estimator: sbs.Estimator,
        x_train: Tensor,
        y_train: Tensor,
    ) -> sbs.SBS:
        "Get the trained sbs algorithm"
        sequential_backward_selection = sbs.SBS(
            estimator=estimator,
            k_features=self.k_features,
            score=accuracy_score,
        )
        sequential_backward_selection.fit(data=x_train, target=y_train)

        return sequential_backward_selection

    def train(self):
        "train data"
        wine_data = self.get_wine_data()
        x_train, x_test, y_train, y_test = prepare_dataset.get_standardized_wine_data()
        estimator = self._get_estimator()
        trained_sbs = self.get_trained_sbs(
            estimator=estimator,
            x_train=x_train,
            y_train=y_train,
        )
        k_feat = [len(k) for k in trained_sbs.subset]
        max_score_at = np.argmax(np.array(trained_sbs.scores[::-1]))
        best = k_feat[::-1][max_score_at]
        features = list(trained_sbs.subset[::-1][max_score_at])
        estimator.fit(
            X=x_train[:, features],
            y=y_train,
        )
        y_pred = estimator.predict(X=x_test[:, features])
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Score: {score} at best: {best}")
        self.plot_sbs(k_feat=k_feat, scores=trained_sbs.scores)
        print(f"{', '.join([wine_data.feature_names[i] for i in features])}")

    def plot_sbs(self, k_feat: List[int], scores: List[float]) -> None:
        "Plot the sbs training data"
        plt.plot(k_feat, scores, marker="o")
        plt.ylim([0.0, 1.02])
        plt.ylabel("Accuracy")
        plt.xlabel("Features")
        plt.grid()
        plt.tight_layout()
        plt.show()
