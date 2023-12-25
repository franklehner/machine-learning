#!/usr/bin/env python
"""Adaline
"""
import click
import matplotlib.pyplot as plt
import numpy as np

from libs.machine_learning.models.perceptron import AdalineGD
from libs.machine_learning.views.iris import Iris


@click.command()
@click.option(
    "--eta",
    default=0.001,
    help="learning rate",
)
@click.option(
    "--iterations",
    default=10,
    help="Iterations",
)
def cli(eta: float, iterations: int):
    """Client
    """
    iris = Iris()
    targets = iris.targets[0:100]
    targets = np.where(targets == "Iris-setosa", -1, 1)
    data = iris.data[0:100, [0, 2]]
    x_std = np.copy(data)
    x_std[:, 0] = (x_std[:, 0] - x_std[:, 0].mean()) / x_std[:, 0].std()
    x_std[:, 1] = (x_std[:, 1] - x_std[:, 1].mean()) / x_std[:, 1].std()
    adaline = AdalineGD(eta=eta, n_iter=iterations)
    adaline.fit(data=x_std, targets=targets)
    plt.plot(
        range(1, len(adaline.costs) + 1),
        np.log10(adaline.costs),
        marker="o",
    )
    plt.xlabel("epochs")
    plt.ylabel("log sum of squared errors")
    plt.title("Adaline")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
