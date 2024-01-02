#!/usr/bin/env python
"""This script ...
"""
import logging
import multiprocessing
from typing import Literal, Optional

import click

from libs.machine_learning.iris.random_forest_classifier import IrisClassifier

_log = logging.getLogger(__name__)
Criterion = Literal["gini", "entropy"]


@click.command()
@click.option(
    "--criterion",
    type=click.Choice(["gini", "entropy"]),
    default="entropy",
    show_choices=True,
    show_default=True,
    help="Which impurity measure should be used?",
)
@click.option(
    "--n_estimators",
    type=click.INT,
    required=False,
    help="How many estimators should be used?",
)
@click.option(
    "--n-jobs",
    type=click.IntRange(min=1, max=multiprocessing.cpu_count()),
    default=multiprocessing.cpu_count(),
    show_choices=True,
    show_default=True,
    help="How many cpu's should be used?",
)
def cli(criterion: Criterion, n_estimators: Optional[int], n_jobs: int):
    """Client
    """
    classifier = IrisClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=1,
    )
    x_train, x_test, y_train, y_test = classifier.get_iris_data()
    classifier.train(x_train=x_train, y_train=y_train)
    y_pred = classifier.predict(data=x_test)
    errors = (y_pred != y_test).sum()
    print(f"Errors: {errors}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
