"""Prepare dataset
"""
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from libs.machine_learning.views.boston_houses import BostonHouses
from libs.machine_learning.views.breast_cancer import BreastCancer
from libs.machine_learning.views.iris import Iris
from libs.machine_learning.views.movie_reviews import Reviews
from libs.machine_learning.views.wine import Wine

Tensor = np.ndarray
SplittedData = Tuple[Tensor, Tensor, Tensor, Tensor]
Name = Literal["iris", "wine", "cancer", "movie", "boston"]
Data = Union[BreastCancer, Iris, Wine, Reviews, BostonHouses]
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

        if self.name == "boston":
            return BostonHouses()

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


def tokenizer(text: str) -> List[str]:
    "Split text into tokens"
    text = re.sub(r"<[^>]*>", "", text)
    emoticons = re.findall(
        r"(?::|;|=)(?:-)?(?:\)|\(|D|P)",
        text.lower(),
    )
    text = re.sub(r"[\W]+", " ", text.lower() + " ".join(emoticons).replace("-", ""))
    tokenized = [word for word in text.split() if word not in STOP]

    return tokenized


def tokenizer_porter(text: str) -> List[str]:
    "Split text into stemmed tokens"
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]
