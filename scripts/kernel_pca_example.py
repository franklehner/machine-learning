#!/usr/bin/env python
"""This script makes an example for kernel pca
"""
import logging
import click
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA


_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    data, targets = make_moons(n_samples=100, random_state=123)
    k_pca = KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=15,
    )
    x_pca = k_pca.fit_transform(data)
    plt.scatter(
        x=x_pca[targets==0, 0],
        y=x_pca[targets==0, 1],
        color="red",
        marker="^",
        alpha=0.5,
    )
    plt.scatter(
        x=x_pca[targets==1, 0],
        y=x_pca[targets==1, 1],
        color="blue",
        marker="o",
        alpha=0.5,
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
