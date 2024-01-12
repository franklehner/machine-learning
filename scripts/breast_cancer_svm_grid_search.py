#!/usr/bin/env python
"""This script classifier the breast cancer data
with an SVM and gridsearch
"""
import logging
from typing import Literal
import click

import libs.machine_learning.cancer.grid_search_svm as gs_svm


_log = logging.getLogger(__name__)
Name = Literal["iris", "wine", "cancer"]


@click.command()
@click.option(
    "--name",
    type=click.Choice(["iris", "wine", "cancer"]),
    default="cancer",
    show_choices=True,
    show_default=True,
    help="Choose database.",
)
@click.option(
    "--test-size",
    "-t",
    type=click.FloatRange(min=0.1, max=0.4),
    default=0.2,
    show_choices=True,
    show_default=True,
    help="Ratio of test data size",
)
@click.option(
    "--cross",
    "-c",
    is_flag=True,
    default=False,
    show_choices=True,
    show_default=True,
    help="Crossvalidate the result",
)
def cli(name: Name, test_size: float, cross: bool):
    """Client
    """
    gs_svm.train(
        name=name,
        test_size=test_size,
        cross=cross,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
