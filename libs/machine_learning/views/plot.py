"""Plot
"""
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

from libs.machine_learning.models.classifier import TreeClassifier

Tensor = np.ndarray
MARKERS = ("s", "x", "o", "^", "v")
COLORS = ("r", "b", "g", "k", "c")


def get_grid(data: Tensor, resolution: float) -> list[Tensor]:
    "Calculate grid by data"
    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    return np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )


def mark_test_data(test_data: Tensor, test_idx: range) -> None:
    "Plot test data"
    x_test = test_data[test_idx, :]
    plt.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c="cyan",
        edgecolors="black",
        alpha=0.2,
        linewidths=1,
        marker="o",
        s=100,
        label="Test data",
    )


def plot_decision_regions(
    data: np.ndarray,
    target: np.ndarray,
    classifier: Any,
    names: Optional[List[str]] = None,
    test_idx: Optional[range] = None,
) -> None:
    """plot region"""
    resolution = 0.02
    xx1, xx2 = get_grid(data=data, resolution=resolution)

    z_value = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z_value = z_value.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z_value, alpha=0.4)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c_map in enumerate(np.unique(target)):
        if names is not None:
            plt.scatter(
                x=data[target == c_map, 0],
                y=data[target == c_map, 1],
                alpha=0.8,
                c=COLORS[idx],
                marker=MARKERS[idx],
                label=names[c_map],
            )
        else:
            plt.scatter(
                x=data[target == c_map, 0],
                y=data[target == c_map, 1],
                alpha=0.8,
                c=COLORS[idx],
                marker=MARKERS[idx],
                label=c_map,
            )

    if test_idx is not None:
        mark_test_data(test_data=data, test_idx=test_idx)


def plot_graph(
    dtree: TreeClassifier,
    class_names: List[str],
    feature_names: List[str],
    filename: str,
) -> None:
    "plot graph"
    dot_data = export_graphviz(
        decision_tree=dtree,
        out_file=None,
        filled=True,
        rounded=True,
        class_names=class_names,
        feature_names=feature_names,
    )
    graph = graph_from_dot_data(data=dot_data)
    graph.write_png(filename)
