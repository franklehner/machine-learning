#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Literal

import click

from libs.machine_learning.iris.decision_tree import IrisClassifier

_log = logging.getLogger(__name__)
Criterion = Literal["gini", "entropy"]


@click.command()
@click.option(
    "--criterion",
    type=click.Choice(["gini", "entropy"]),
    default="entropy",
    show_default=True,
    show_choices=True,
    help="What impurity should be used (gini, entropy)?",
)
@click.option(
    "--max-depth",
    type=click.INT,
    default=4,
    show_default=True,
    help="At which count of shapes should be pruned?",

)
@click.option(
    "--filename",
    type=click.STRING,
    default="data/tree.png",
    show_default=True,
    help="Name of tree graph file",
)
@click.option(
    "--graph",
    is_flag=True,
    default=False,
    show_default=True,
    show_choices=True,
    help="Write graph",
)
def cli(criterion: Criterion, max_depth: int, filename: str, graph: bool):
    """Client
    """
    classifier = IrisClassifier(
        criterion=criterion,
        max_depth=max_depth,
        filename=filename,
        random_state=1,
    )
    x_train, x_test, y_train, y_test = classifier.get_iris_data()
    dtree = classifier.init_decision_tree()
    dtree = classifier.train(x_train=x_train, y_train=y_train, tree=dtree)
    y_pred = classifier.predict(data=x_test, tree=dtree)
    errors = (y_pred != y_test).sum()
    print(f"Errors: {errors}")
    if graph:
        classifier.plot_graph(
            tree=dtree,
            class_names=classifier.class_names,
            feature_names=classifier.feature_names,
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
