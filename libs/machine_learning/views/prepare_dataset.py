"""Prepare dataset
"""
from typing import Optional, Tuple

import numpy as np

from libs.machine_learning.views.iris import Iris

Tensor = np.ndarray
SplittedData = Tuple[Tensor, Tensor, Tensor, Tensor]


def get_origin_iris_dataset() -> Iris:
    "get the dataset without restriction"
    return Iris()


def get_splitted_dataset(iris: Optional[Iris] = None, count: Optional[int] = None) -> SplittedData:
    "get splitted dataset"
    if iris is None:
        iris = get_origin_iris_dataset()
    if count is None:
        x_train, x_test, y_train, y_test = iris.split()
    else:
        x_train, x_test, y_train, y_test = iris.split(count=count)

    return x_train, x_test, y_train, y_test


def get_standardized_data(count: Optional[int] = None) -> SplittedData:
    "Get dataset with mean 0 and std 1"
    iris = get_origin_iris_dataset()
    x_train, x_test, y_train, y_test = get_splitted_dataset(
        iris=iris,
        count=count,
    )
    x_train, x_test = iris.standardize(
        data_train=x_train, data_test=x_test,
    )

    return x_train, x_test, y_train, y_test
