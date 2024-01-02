#!/usr/bin/env python
"""Load iris dataset and plot
"""
import logging

import click
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

from libs.machine_learning.views.iris import Iris
from libs.machine_learning.views.plot import plot_decision_regions

_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    iris = Iris()
    targets = iris.targets
    data = iris.data[:, [2, 3]]
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.3, random_state=0,
    )
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    ppn = Perceptron(eta0=0.1, verbose=1)
    ppn.fit(x_train_std, y_train)
    y_pred = ppn.predict(x_test_std)
    print(f"False classified = {(y_test != y_pred).sum()}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    plot_decision_regions(data=x_train_std, target=y_train, classifier=ppn)
    plt.xlabel("Länge des Kelchblatts [cm]")
    plt.ylabel("Länge des Blütenblatts [cm]")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
