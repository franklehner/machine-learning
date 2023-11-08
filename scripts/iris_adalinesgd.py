#!/usr/bin/env python
"""Adaline with stochastic gradient descent
"""
import logging

import click
import matplotlib.pyplot as plt
import numpy as np

from libs.machine_learning.models.perceptron import AdalineSGD
from libs.machine_learning.views.iris import Iris
from libs.machine_learning.views.plot import plot_decision_regions

_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--eta",
    default=0.01,
    help="Learning rate",
)
@click.option(
    "--n-iter",
    default=15,
    help="Number of iterations",
)
@click.option(
    "--random-state",
    default=1,
    help="Random state for repeating",
)
def cli(eta: float, n_iter: int, random_state: int):
    """Client"""
    iris = Iris()
    targets = iris.targets[0:100]
    targets = np.where(targets == "Iris-setosa", -1, 1)
    data = iris.data[0:100, [0, 2]]
    ada = AdalineSGD(eta=eta, n_iter=n_iter, random_state=random_state)
    ada.fit(data=data, targets=targets)
    plot_decision_regions(data=data, target=targets, classifier=ada)
    plt.xlabel("Länge des Kelchblatts [cm]")
    plt.ylabel("Länge des Blütenblatts [cm]")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
