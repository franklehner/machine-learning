#!/usr/bin/env python
"""This script uses ensemble learning
"""
import logging
import click

from libs.machine_learning import ensemble_learning as ensemble


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--vote",
    type=click.Choice(["classlabel", "probability"]),
    default="classlabel",
    show_choices=True,
    show_default=True,
    help="How to vote?",
)
@click.option(
    "--test-size",
    "-t",
    type=click.FloatRange(min=0.1, max=0.9),
    default=0.5,
    show_choices=True,
    show_default=True,
    help="Ratio of the test size",
)
@click.option(
    "--stratify",
    "-s",
    is_flag=True,
    default=False,
    show_choices=True,
    show_default=True,
    help="Split by stratified labels",
)
def cli(vote: ensemble.Vote, test_size: float, stratify: bool):
    """Client
    """
    learner = ensemble.EnsembleLearning(
        vote=vote,
        test_size=test_size,
        stratify=stratify,
    )
    learner.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
