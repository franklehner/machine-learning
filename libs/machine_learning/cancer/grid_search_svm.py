"GridSearch with SVM"
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from libs.machine_learning.views import prepare_dataset


Name = Literal["iris", "wine", "cancer"]
Tensor = np.ndarray


def gridsearch(random_state: int = 1, cv: int = 10) -> GridSearchCV:
    "gridsearch"
    pipe_svc = make_pipeline(
        StandardScaler(),
        SVC(random_state=random_state),
    )
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [
        {
            "svc__C": param_range,
            "svc__kernel": ["linear"],
        },
        {
            "svc__C": param_range,
            "svc__gamma": param_range,
            "svc__kernel": ["rbf"],
        },
    ]
    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        refit=True,
        n_jobs=-1,
    )

    return gs


def get_dataset(name: Name) -> prepare_dataset.Data:
    "get the data"
    dataset = prepare_dataset.DataSet(name=name)

    return dataset.get_dataset()


def train(name: Name, test_size: float = 0.2, cross: bool = False) -> None:
    "train"
    data = get_dataset(name=name)
    x_train, x_test, y_train, y_test = data.split(
        test_size=test_size,
        stratify=True,
    )
    if cross:
        estimator = gridsearch(cv=2)
        scores = cross_val_score(
            estimator=estimator,
            X=x_train,
            y=y_train,
            scoring="accuracy",
            cv=5,
        )
        print(f"KV score: {np.mean(scores)} +/- {np.std(scores)}")
    else:
        estimator = gridsearch()
    estimator.fit(X=x_train, y=y_train)
    print(f"Best score: {estimator.best_score_}")
    print(f"Best params: {estimator.best_params_}")
    classifier = estimator.best_estimator_
    classifier.fit(x_train, y_train)
    print(f"Test Accuracy: {classifier.score(x_test, y_test)}")
    y_pred = classifier.predict(x_test)
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred)


def plot_confusion_matrix(y_true: Tensor, y_pred: Tensor) -> None:
    "plot confusion_matrix"
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    _, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, alpha=0.3) # pylint: disable=no-member
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.show()
    print(f"Accuracy: {precision_score(y_true=y_true, y_pred=y_pred)}")
    print(f"Recall: {recall_score(y_true=y_true, y_pred=y_pred)}")
    print(f"F1: {f1_score(y_true=y_true, y_pred=y_pred)}")
