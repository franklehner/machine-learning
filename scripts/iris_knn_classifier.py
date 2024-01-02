#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Literal

import click

from libs.machine_learning.iris.knn import IrisClassifier

_log = logging.getLogger(__name__)
POW = Literal[1, 2]


@click.command()
@click.option(
    "--n-neighbors",
    type=click.INT,
    default=5,
    show_default=True,
    help="How many neighbors should be used for classification?",
)
@click.option(
    "--p-param",
    type=click.IntRange(min=1, max=2),
    default=2,
    show_choices=True,
    show_default=True,
    help="Which metric should be used (manhattan: 1, euclid: 2)?",
)
def cli(n_neighbors: int, p_param: POW):
    """Client
    """
    print(f"Neighbors: {n_neighbors}")
    print(f"P-Param: {p_param}")
    knn = IrisClassifier(
        n_neighbors=n_neighbors,
        p_param=p_param,
    )
    x_train, x_test, y_train, y_test = knn.get_iris_data()
    knn.train(x_train=x_train, y_train=y_train)
    y_pred = knn.predict(data=x_test)
    errors = (y_pred != y_test).sum()
    print(f"Errors: {errors}")



if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
