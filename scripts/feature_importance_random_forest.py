#!/usr/bin/env python
"""This script shows the most important features of wine dataset
"""
import logging
import click
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import SelectFromModel

from libs.machine_learning.views import prepare_dataset
from libs.machine_learning.models import sbs


_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    wine_data = prepare_dataset.get_origin_wine_dataset()
    x_train, _, y_train, _ = prepare_dataset.get_splitted_wine_dataset()
    feature_labels = wine_data.feature_names
    forest = sbs.RandomForestClassifier(
        n_estimators=500, random_state=1,
    )
    forest.fit(X=x_train, y=y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print(f"({f + 1}) {feature_labels[indices[f]]} {importances[indices[f]]}")

    plt.title("Meaning of features")
    plt.bar(range(x_train.shape[1]), importances[indices], align="center")
    plt.xticks(
        range(x_train.shape[1]),
        labels=np.array(feature_labels)[indices],
        rotation=90,
    )
    plt.xlim([-1, x_train.shape[1]])
    plt.tight_layout()
    plt.show()
    sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
    x_selected = sfm.transform(x_train)
    print()
    for f in range(x_selected.shape[1]):
        print(f"({f + 1}) {feature_labels[indices[f]]} - {importances[indices[f]]}")




if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
