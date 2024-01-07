#!/usr/bin/env python
"""This script classifies wine data with kernel pca
"""
import logging
from typing import Literal, Optional

import click

from libs.machine_learning.wine.logistic_regression import WineClassifier


_log = logging.getLogger(__name__)
MultiClass = Literal["ovr", "multinomial"]
Solver = Literal["linear", "lbfgs"]
Kernel = Literal["linear", "rbf"]


@click.command()
@click.option(
    "--n-components",
    "-n",
    type=click.INT,
    required=False,
    help="Number of components",
)
@click.option(
    "--multiclass",
    "-m",
    type=click.Choice(["ovr", "multinomial"]),
    default="ovr",
    show_choices=True,
    show_default=True,
    help="How to work with multi classes",
)
@click.option(
    "--solver",
    "-s",
    type=click.Choice(["linear", "lbfgs"]),
    default="lbfgs",
    show_choices=True,
    show_default=True,
    help="Which solver should be used?",
)
@click.option(
    "--kernel",
    "-k",
    type=click.Choice(["linear", "rbf"]),
    default="rbf",
    show_choices=True,
    show_default=True,
    help="What kind of kernel should be used?",
)
@click.option(
    "--gamma",
    "-g",
    type=click.FLOAT,
    default=15.0,
    show_choices=True,
    show_default=True,
    help="Value of gamma.",
)
def cli(
    n_components: Optional[int],
    multiclass: MultiClass,
    solver: Solver,
    kernel: Kernel,
    gamma: float,
):
    """Client"""
    wine_classifier = WineClassifier(
        multiclass=multiclass,
        solver=solver,
        reduction="kernel-pca",
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
    )
    wine_classifier.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
