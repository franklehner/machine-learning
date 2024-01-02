#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Literal, Optional, Tuple

import click
import numpy as np

from libs.machine_learning.iris.support_vector_machine import (
    FeaturesIndex,
    IrisClassifier,
)

_log = logging.getLogger(__name__)
Label = Literal["petal", "sepal"]
Labels = Tuple[Label, ...]
Attribute = Literal["width", "length"]
Attributes = Tuple[Attribute, ...]
Kernel = Literal["linear", "rbf"]


@click.command()
@click.option(
    "--kernel",
    type=click.STRING,
    default="linear",
    show_default=True,
    help="Kernel type of the support vector machine (linear, rbf)",
)
@click.option(
    "--regularize",
    "-r",
    type=click.FLOAT,
    default=1.0,
    show_default=True,
    help="Regularize the errors",
)
@click.option(
    "--gamma",
    type=click.FLOAT,
    default=0.1,
    show_default=True,
    help="regularize data",
)
@click.option(
    "--label",
    type=click.STRING,
    required=False,
    multiple=True,
    help="Train only on this label (petal, sepal)",
)
@click.option(
    "--attribute",
    type=click.STRING,
    required=False,
    multiple=True,
    help="Train only this attribute (width, length)",
)
def cli(
    kernel: Kernel,
    regularize: float,
    gamma: float,
    label: Optional[Labels],
    attribute: Optional[Attributes],
):
    """Client"""
    indexes = FeaturesIndex(labels=label, attributes=attribute).indexes
    classifier = IrisClassifier(indexes=indexes)
    svm = classifier.init_support_vector_machine(
        regularize=regularize,
        kernel=kernel,
        random_state=1,
        gamma=gamma,
    )
    x_train, x_test, y_train, y_test = classifier.get_iris_dataset()
    classifier.train(x_train=x_train, y_train=y_train, svm=svm)
    x_combined_std = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    names = [str(name) for name in classifier.target_names]
    if len(indexes) == 2:
        classifier.plot(
            svm=svm,
            data=x_combined_std,
            target=y_combined,
            names=names,
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
