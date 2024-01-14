"""Prepare dataset
"""
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from libs.machine_learning.views.breast_cancer import BreastCancer
from libs.machine_learning.views.iris import Iris
from libs.machine_learning.views.movie_reviews import Reviews
from libs.machine_learning.views.wine import Wine

Tensor = np.ndarray
SplittedData = Tuple[Tensor, Tensor, Tensor, Tensor]
Name = Literal["iris", "wine", "cancer", "movie"]
Data = Union[BreastCancer, Iris, Wine, Reviews]
STOP = stopwords.words("english")


@dataclass
class DataSet:
    "Dataset"

    name: Name

    def get_dataset(self) -> Data:
        "get the dataset"
        if self.name == "iris":
            return Iris()

        if self.name == "wine":
            return Wine()

        if self.name == "movie":
            return Reviews()

        return BreastCancer()


def get_origin_iris_dataset() -> Iris:
    "get the dataset without restriction"
    return Iris()

def get_origin_wine_dataset() -> Wine:
    "Get the wine dataset without restriction"
    return Wine()


def get_origin_breast_cancer_dataset() -> BreastCancer:
    "Get the origin datast of breast cancer"
    return BreastCancer()


def get_splitted_dataset(iris: Optional[Iris] = None, count: Optional[int] = None) -> SplittedData:
    "get splitted dataset"
    if iris is None:
        iris = get_origin_iris_dataset()
    if count is None:
        x_train, x_test, y_train, y_test = iris.split()
    else:
        x_train, x_test, y_train, y_test = iris.split(count=count)

    return x_train, x_test, y_train, y_test

def get_splitted_wine_dataset() -> SplittedData:
    "Get splitted wine dataset"
    wine = get_origin_wine_dataset()
    return wine.split()


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

def get_standardized_wine_data() -> SplittedData:
    "Get standardized wine dataset with mean 0 and std 1"
    wine = get_origin_wine_dataset()
    x_train, x_test, y_train, y_test = wine.split()
    x_train_std, x_test_std = wine.standardize(
        x_train=x_train,
        x_test=x_test,
    )

    return x_train_std, x_test_std, y_train, y_test
