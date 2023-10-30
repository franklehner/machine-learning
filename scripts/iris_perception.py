#!/usr/bin/env python
"""Load iris dataset and plot
"""
import logging

import click
import matplotlib.pyplot as plt
import numpy as np

from libs.machine_learning.models.perceptron import Perceptron
from libs.machine_learning.views.plot import plot_decision_regions
from libs.machine_learning.views.iris import Iris

_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    iris = Iris()
    targets = iris.targets[0:100]
    targets = np.where(targets == "Iris-setosa", -1, 1)
    data = iris.data[0:100, [0, 2]]
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(data, targets)

    plot_decision_regions(data=data, target=targets, classifier=ppn)
    plt.xlabel("Länge des Kelchblatts [cm]")
    plt.ylabel("Länge des Blütenblatts [cm]")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
