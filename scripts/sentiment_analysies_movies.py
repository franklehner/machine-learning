#!/usr/bin/env python
"""This script ...
"""
import logging
import os
import pickle
from typing import Dict, Generator, List, Optional, Tuple

import click
import numpy as np
import pyprind
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier

from libs.machine_learning.reviews.vectorizer import get_vectorizer

_log = logging.getLogger(__name__)
STOP = stopwords.words("english")
Stream = Generator[Tuple[str, int], None, None]
Batch = Tuple[Optional[List[str]], Optional[List[int]]]


def stream_docs(path: str) -> Stream:
    "stream data"
    with open(path, "r", encoding="utf-8") as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream: Stream, size: int) -> Batch:
    "get minibatch"
    docs, labels = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            labels.append(label)
    except StopIteration:
        return None, None

    return docs, labels


@click.group()
@click.pass_context
@click.option(
    "--dest",
    "-d",
    type=click.STRING,
    default="data/movieclassifier/pkl_objects",
    show_default=True,
    help="Path to the model",
)
def cli(ctx, dest: int):
    """Client"""
    ctx.obj = {"dest": dest}


@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.STRING,
    default="data/movie_data.csv",
    show_default=True,
    help="Path to the reviews",
)
@click.pass_obj
def train(ctx: Dict[str, str], path: str):
    "Train data"
    dest = ctx["dest"]
    vect = get_vectorizer()
    clf = SGDClassifier(
        loss="log_loss",
        random_state=1,
    )
    doc_stream = stream_docs(path=path)
    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        x_train, y_train = get_minibatch(doc_stream=doc_stream, size=1000)
        if x_train is None:
            break
        x_train = vect.transform(x_train)
        clf.partial_fit(x_train, y_train, classes=classes)
        pbar.update()

    x_test, y_test = get_minibatch(doc_stream=doc_stream, size=5000)
    x_test = vect.transform(x_test)
    print(f"Test Accuracy: {clf.score(x_test, y_test)}")
    clf.partial_fit(x_test, y_test)
    if not os.path.exists(dest):
        os.makedirs(dest)

    with open(os.path.join(dest, "stopwords.pkl"), "wb") as fobj:
        pickle.dump(STOP, fobj, protocol=4)

    with open(os.path.join(dest, "classifier.pkl"), "wb") as fobj:
        pickle.dump(clf, fobj, protocol=4)


@cli.command()
@click.pass_obj
def load(ctx: Dict[str, str]):
    "Load data"
    print(ctx)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
