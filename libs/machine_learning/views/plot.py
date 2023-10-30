"""Plot
"""
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(
    data: np.ndarray, target: np.ndarray, classifier: Any, resolution: float = 0.02,
):
    """plot region
    """
    markers = tuple("sxo^v")
    colors = (
        "r",
        "b",
        "g",
        "k",
        "c",
    )
    # cmap = ListedColormap(colors[:len(np.unique(target))])

    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    z_value = classifier.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T
    )
    z_value = z_value.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z_value, alpha=0.4, colormap=colors)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c_map in enumerate(np.unique(target)):
        plt.scatter(
            x=data[target == c_map, 0],
            y=data[target == c_map, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=c_map,
        )
