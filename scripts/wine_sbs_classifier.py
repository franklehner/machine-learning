#!/usr/bin/env python
"""This script ...
"""
import logging
import click
from libs.machine_learning.wine import sbs_classifier


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--estimator",
    type=click.Choice(
        ["knn", "svm", "tree", "logres", "random_forest"],
    ),
    default="knn",
    show_choices=True,
    show_default=True,
    help="Which classifier should be used?",
)
@click.option(
    "--k-features",
    type=click.INT,
    default=1,
    show_default=True,
    help="How many features should be used?",
)
def cli(estimator: sbs_classifier.ClassifierNames, k_features: int):
    """Client
    """
    wine = sbs_classifier.WineClassifier(
        classifier_name=estimator,
        k_features=k_features,
    )
    wine.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
