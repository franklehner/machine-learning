#!/usr/bin/env python
"""This script ...
"""
import logging
import click

from libs.machine_learning.reviews.sentiment_classifier import SentimentAnalysis


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.STRING,
    default="data/movie_data.parquet",
    show_choices=True,
    show_default=True,
    help="Full file path to movie data",
)
@click.option(
    "--stem",
    "-s",
    is_flag=True,
    default=False,
    help="Use stem",
)
def cli(path: str, stem: bool):
    """Client
    """
    sentiment_analysis = SentimentAnalysis(
        path=path,
        stem=stem,
    )
    sentiment_analysis.train()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
