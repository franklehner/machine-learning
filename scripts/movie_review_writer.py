#!/usr/bin/env python
"""This script reads the extracted tarfile and writes it to parquet
"""
import logging
import os

import click
import numpy as np
import pandas as pd
import pyprind


_log = logging.getLogger(__name__)


def get_data(basepath: str) -> pd.DataFrame:
    "get movie data"
    pbar = pyprind.ProgBar(50000)
    labels = {"pos": 1, "neg": 0}
    dfs = []
    for sub in ("test", "train"):
        for label in ("pos", "neg"):
            path = os.path.join(basepath, sub, label)
            for file in os.listdir(path):
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    txt = infile.read()
                dfs.append(pd.DataFrame([[txt, labels[label]]]))
                pbar.update()

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={0: "reviews", 1: "sentiment"}, inplace=True)
    df = df.reindex(np.random.permutation(df.index))

    return df


@click.command()
@click.option(
    "--basepath",
    "-b",
    type=click.STRING,
    default="aclImdb",
    show_choices=True,
    show_default=True,
    help="Path to the movie data",
)
@click.option(
    "--outfile",
    "-o",
    type=click.STRING,
    default="data/movie_data.parquet",
    show_choices=True,
    show_default=True,
    help="Path to the parquet file",
)
def cli(basepath: str, outfile: str):
    """Client
    """
    data = get_data(basepath=basepath)
    data.to_parquet(outfile, index=False)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
