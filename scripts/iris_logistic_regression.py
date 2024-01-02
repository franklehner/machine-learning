#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Literal

import click
import numpy as np

from libs.machine_learning.iris.logistic_regression import FeaturesIndex, IrisClassifier

Label = Literal["petal", "sepal"]
MultiClass = Literal[
    "auto",
    "ovr",
    "multinomial",
]


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--label",
    type=click.STRING,
    default="petal",
    help="Train by label",
)
@click.option(
    "--standardize",
    "-s",
    is_flag=True,
    default=False,
    help="Standardize data?",
)
@click.option(
    "--multi-class", type=click.STRING, default="multinomial", help="How to train data?"
)
def cli(label: Label, standardize: bool, multi_class: MultiClass):
    """Client"""
    features = FeaturesIndex(
        labels=[label],
    )
    ic = IrisClassifier(
        features=features,
        standardize=standardize,
    )
    lrc = ic.init_logistic_regression(multi_class=multi_class)
    x_train, x_test, y_train, y_test = ic.get_iris_dataset()
    ic.train(x_train=x_train, y_train=y_train, lrc=lrc)
    x_combined_std = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    names = [str(name) for name in ic.target_names]
    ic.plot(lrc=lrc, data=x_combined_std, target=y_combined, names=names)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
