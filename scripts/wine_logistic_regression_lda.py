#!/usr/bin/env python
"""This script classifies wine data with logistic LogisticRegression
and LDA
"""
import logging
from typing import Literal, Optional

import click

from libs.machine_learning.wine.logistic_regression import WineClassifier


_log = logging.getLogger(__name__)
MultiClass = Literal["ovr", "multinomial"]
Solver = Literal["linear", "lbfgs"]


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
def cli(n_components: Optional[int], multiclass: MultiClass, solver: Solver):
    """Client
    """
    wine_classifier = WineClassifier(
        multiclass=multiclass,
        solver=solver,
        reduction="lda",
        n_components=n_components,
    )
    wine_classifier.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
