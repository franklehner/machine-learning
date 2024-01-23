"""Classifier, predictor and explorer for Boston house prices
"""
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    RANSACRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from libs.machine_learning.models.regression import LinearRegressionGD
from libs.machine_learning.views.boston_houses import BostonHouses


Cols = Literal[
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
    "medv",
]
Pivot = Tuple[Cols, ...]
Regressor = Literal[
    "linear",
    "ransac",
    "lineargd",
    "elastic",
    "forest",
    "polynom",
    "random",
]
Model = Union[
    LinearRegression,
    RANSACRegressor,
    PolynomialFeatures,
    DecisionTreeRegressor,
    LinearRegressionGD,
    Pipeline,
    ElasticNet,
    RandomForestRegressor,
]
Tensor = np.ndarray
TrainTestData = Tuple[Tensor, Tensor, Tensor, Tensor]
Loss = Literal["absolute_error", "squared_error"]
RFCriterion = Literal[
    "squared_error",
    "absolute_error",
    "friedman_mse",
    "poisson",
]
Kernels = Literal[1, 2, 3, 4, 5, 6, 7, 8]


REGRESSOR_MAPPER = {
    "linear": LinearRegression,
    "ransac": RANSACRegressor,
    "forest": DecisionTreeRegressor,
    "polynom": PolynomialFeatures,
    "lineargd": LinearRegressionGD,
    "elastic": ElasticNet,
    "random": RandomForestRegressor,
}


@dataclass
class ElasticNetParams:
    "Set parameters for L2 regularization"

    alpha: float = 1.0
    l1_ratio: float = 0.5


@dataclass
class PolynomParams:
    "Set parameters for PolynomialFeatures"

    degree: int = 2


@dataclass
class LinearRegressionGDParams:
    "Set Params for the Linear regression gd"

    eta: float = 0.001
    n_iter: int = 20


@dataclass
class RANSACRegressorParams:
    "Set Params for RANSAC"

    max_trials: int = 100
    min_samples: int = 50
    loss: Loss = "absolute_error"
    residual_threshold: float = 5.0
    random_state: int = 0


@dataclass
class DecisionTreeParams:
    "Set params for decision tree"

    max_depth: int = 3


@dataclass
class RandomForestParams:
    "Set parameters for RandomForestRegressor"

    n_estimators: int = 1000
    criterion: RFCriterion = "squared_error"
    random_state: int = 1
    n_jobs: Kernels = 8


@dataclass
class Explorer:
    "Explorer for Boston house prices"

    pivot: Pivot
    save: bool
    hist: bool
    columns: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.pivot:
            self.columns = [pivot.upper() for pivot in self.pivot]

    def load(self) -> pd.DataFrame:
        "Load boston house data"
        boston_houses = BostonHouses()

        if self.columns:
            return boston_houses.data[self.columns]

        return boston_houses.data

    def plot(self, data: pd.DataFrame) -> None:
        "Plot a scatter matrix"
        pd.plotting.scatter_matrix(data)
        if self.save:
            plt.savefig("data/boston/graphs/scatter_matrix.png")
        else:
            plt.show()

        if self.hist:
            self.histogram(data=data)

    def correlation(self, data: pd.DataFrame) -> None:
        "Show correlation of the data"
        print(f"\nComplete correlation:\n{data.corr()}")
        if self.pivot:
            for col in self.pivot:
                print(
                    f"\nCorrelation with {col.upper()}:\n{data.corrwith(data[col.upper()])}"
                )

    def histogram(self, data: pd.DataFrame) -> None:
        "Show histogram"
        for column in data.columns:
            data[column].hist(legend=True)
            if self.save:
                plt.savefig(f"data/boston/graphs/{column}_hist.png")
            else:
                plt.show()


Parameters = Union[
    LinearRegressionGDParams,
    RANSACRegressorParams,
    ElasticNetParams,
    PolynomParams,
    DecisionTreeParams,
    RandomForestParams,
]


@dataclass
class RegressionModel:
    "Build the model for the regression"

    regressor: Regressor
    model: Model = field(init=False)
    parameters: Optional[Parameters] = None
    standardize: bool = False

    def __post_init__(self):
        if self.regressor in ("random", "forest"):
            self.standardize = False

        if self.regressor == "polynom":
            self.model = REGRESSOR_MAPPER["linear"]
            self.model = self.model()
        else:
            self.model = REGRESSOR_MAPPER[self.regressor]
            if self.parameters is not None:
                params = {
                    key: value
                    for key, value in asdict(self.parameters).items()
                    if value is not None
                }
                self.model = self.model(**params)
            else:
                self.model = self.model()

        if self.standardize:
            sc = StandardScaler()
            self.model = Pipeline(
                [
                    ("scaler", sc),
                    ("clf", self.model),
                ],
            )


@dataclass
class Trainer:
    "Trainer for regression model"

    model: RegressionModel
    pivot: Cols
    model_name: str
    test_size: float = 0.2

    def load(self) -> pd.DataFrame:
        "Load boston house data"
        boston_houses = BostonHouses()

        return boston_houses.data

    def get_data(self) -> TrainTestData:
        "Get training and test data"
        boston_houses = self.load()
        columns = [
            column for column in boston_houses.columns if column != self.pivot.upper()
        ]
        x_data = boston_houses[columns].values
        y_data = boston_houses[self.pivot.upper()].values
        return train_test_split(
            x_data,
            y_data,
            test_size=self.test_size,
            random_state=1,
        )

    def evaluate(
        self,
        y_train: Tensor,
        y_train_pred: Tensor,
        y_test: Tensor,
        y_test_pred: Tensor,
    ) -> None:
        "Evaluate model"
        plt.scatter(
            y_train_pred,
            y_train_pred - y_train,
            c="steelblue",
            marker="o",
            edgecolors="white",
            label="Training data",
        )
        plt.scatter(
            y_test_pred,
            y_test_pred - y_test,
            c="limegreen",
            marker="s",
            edgecolors="white",
            label="Test data",
        )
        plt.title(self.model.regressor)
        plt.xlabel("predicted values")
        plt.ylabel("Residuals")
        plt.legend(loc="upper left")
        plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color="black")
        plt.xlim([-10, 50])
        plt.show()
        print(f"Regression with {self.model.regressor}")
        print(
            f"MSE-training: {mean_squared_error(y_train, y_train_pred)}",
            f"MSE-test: {mean_squared_error(y_test, y_test_pred)}",
        )
        print(
            f"R^2 training: {r2_score(y_train, y_train_pred)}",
            f"R^2 test: {r2_score(y_test, y_test_pred)}",
        )

    def train(self) -> None:
        "train regression"
        x_train, x_test, y_train, y_test = self.get_data()
        if self.model.regressor == "polynom":
            if self.model.parameters is not None:
                polynom = PolynomialFeatures(
                    degree=asdict(self.model.parameters).get("degree", 2),
                )
                x_train = polynom.fit_transform(x_train)
                x_test = polynom.fit_transform(x_test)
        self.model.model.fit(x_train, y_train)
        y_train_pred = self.model.model.predict(x_train)
        y_test_pred = self.model.model.predict(x_test)
        self.evaluate(
            y_train_pred=y_train_pred,
            y_train=y_train,
            y_test=y_test,
            y_test_pred=y_test_pred,
        )
