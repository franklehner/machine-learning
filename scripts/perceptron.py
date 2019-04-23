#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perceptron
"""


import click as _click
import libs.app.perceptron as _app


@_click.command()
@_click.option(
    "--url",
    default="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)
def cli(url):
    print(url)


if __name__ == "__main__":
    cli()
