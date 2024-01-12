#!/usr/bin/env python
"""This script classifies the breast cancer data
"""
import logging
from typing import Optional
import click

from libs.machine_learning.cancer import logistic_regression


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--solver",
    "-s",
    type=click.Choice(["linear", "lbfgs"]),
    default="lbfgs",

    show_choices=True,
    show_default=True,
    help="Solver for the logistic regression",
)
@click.option(
    "--n-components",
    "-n",
    type=click.INT,
    required=False,
    help="How many features after dimension reduction should be used?",
)
def cli(solver: logistic_regression.Solver, n_components: Optional[int]):
    """Client
    """
    cancer = logistic_regression.BreastCancer(
        n_components=n_components,
        solver=solver,
    )
    cancer.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
