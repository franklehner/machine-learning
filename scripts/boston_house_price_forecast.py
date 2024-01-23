#!/usr/bin/env python
"""This script makes a forecast for house prices in Boston in the 70's
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import click
from libs.machine_learning.boston_house import boston_house

_log = logging.getLogger(__name__)
COLUMNS = [
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
REGRESSORS = [
    "linear",
    "ransac",
    "forest",
    "polynom",
    "lineargd",
    "elastic",
    "random",
]
CRITERIONS = [
    "absolute_error",
    "squared_error",
    "friedman_mse",
    "poisson",
]
LOSSES = ["absolute_error", "squared_error"]
DEFAULT_LINEAR_REGRESSOR = "linear"
DEFAULT_REGRESSOR = "linear"
DEFAULT_MODEL = "data/boston/pkl_objects/{regressor}_regression_{pivot}.pkl"
DEFAULT_PIVOT = "medv"


@dataclass
class TrainContext:
    "Parse Parameters"

    model: str
    test_size: float
    pivot: Optional[boston_house.Cols] = None
    graph: List[str] = field(default_factory=list)


@click.group(chain=True)
@click.pass_context
@click.option(
    "--model",
    "-m",
    type=click.STRING,
    default=DEFAULT_MODEL,
    show_default=True,
    help="Path to the model",
)
@click.option(
    "--test-size",
    type=click.FloatRange(min=0.1, max=0.9),
    default=0.2,
    show_choices=True,
    show_default=True,
    help="Ratio between training and test data",
)
def cli(ctx, model: str, test_size: float):
    "Client"
    ctx.obj = TrainContext(
        model=model,
        test_size=test_size,
        graph=["client"],
    )


@cli.command("train")
@click.pass_obj
@click.option(
    "--pivot",
    "-p",
    type=click.Choice(COLUMNS),
    default=DEFAULT_PIVOT,
    show_choices=True,
    show_default=True,
    help="Which column should be predicted?",
)
def train(
    ctx: TrainContext,
    pivot: boston_house.Cols,
):
    "Train the model"
    ctx.pivot = pivot
    ctx.graph.append("train")


@cli.command("linear-regression")
@click.pass_obj
@click.option(
    "--standardize",
    is_flag=True,
    default=False,
    help="Should data get standardized?",
)
def linear_regression(
    ctx: TrainContext,
    standardize: bool,
):
    "Linear regression"
    regressor: boston_house.Regressor = "linear"
    ctx.graph.append("linear")
    model = boston_house.RegressionModel(
        regressor=regressor,
        standardize=standardize,
    )
    if ctx.pivot is None:
        ctx.pivot = "medv"

    trainer = boston_house.Trainer(
        model=model,
        pivot=ctx.pivot,
        model_name=ctx.model.format(regressor=regressor, pivot=ctx.pivot),
        test_size=ctx.test_size,
    )
    trainer.train()


@cli.command("elastic-net")
@click.pass_obj
@click.option(
    "--standardize",
    is_flag=True,
    default=False,
    help="Should data get standardized?",
)
@click.option(
    "--alpha",
    type=click.FLOAT,
    default=1.0,
    show_choices=True,
    show_default=True,
    help="Higher alpha means a stronger L2 regularization",
)
@click.option(
    "--l1-ratio",
    type=click.FLOAT,
    default=0.5,
    show_default=True,
    help="Higher l1-ration means a stronger l1 regularization",
)
def elastic_net(ctx: TrainContext, standardize: bool, alpha: float, l1_ratio: float):
    "Logistic regression with L2-regularization"
    ctx.graph.append("elastic")
    regressor: boston_house.Regressor = "elastic"
    params = boston_house.ElasticNetParams(alpha=alpha, l1_ratio=l1_ratio)
    model = boston_house.RegressionModel(
        regressor=regressor,
        parameters=params,
        standardize=standardize,
    )
    if ctx.pivot is None:
        ctx.pivot = "medv"

    trainer = boston_house.Trainer(
        model=model,
        pivot=ctx.pivot,
        model_name=ctx.model.format(regressor=regressor, pivot=ctx.pivot),
        test_size=ctx.test_size,
    )
    trainer.train()


@cli.command("polynom")
@click.pass_obj
@click.option(
    "--standardize",
    is_flag=True,
    default=False,
    help="Should data get standardized",
)
@click.option(
    "--degree",
    type=click.IntRange(min=1, max=100),
    default=2,
    show_choices=True,
    show_default=True,
    help="Degree for the function curve",
)
def polynom(ctx: TrainContext, standardize: bool, degree: int):
    "Linear Regression with polynomial features"
    ctx.graph.append("polynom")
    regressor: boston_house.Regressor = "polynom"
    ctx.pivot = "medv" if ctx.pivot is None else ctx.pivot
    model_name = ctx.model.format(regressor=regressor, pivot=ctx.pivot)
    parameters = boston_house.PolynomParams(degree=degree)
    model = boston_house.RegressionModel(
        regressor=regressor,
        parameters=parameters,
        standardize=standardize,
    )
    trainer = boston_house.Trainer(
        model=model,
        pivot=ctx.pivot,
        model_name=model_name,
        test_size=ctx.test_size,
    )
    trainer.train()


@cli.command("ransac")
@click.pass_obj
@click.option(
    "--max-trials",
    type=click.INT,
    default=100,
    show_default=True,
    help="Set the number of max iterations",
)
@click.option(
    "--min-samples",
    type=click.INT,
    default=50,
    show_default=True,
    help="Count of samples which are used (minimum)",
)
@click.option(
    "--loss",
    type=click.Choice(LOSSES),
    default="absolute_error",
    show_choices=True,
    show_default=True,
    help="Which loss function should be used?",
)
@click.option(
    "--threshold",
    type=click.FLOAT,
    default=5.0,
    show_default=True,
    help="Maximal distance of the residuals.",
)
def ransac(
    ctx: TrainContext,
    max_trials: int,
    min_samples: int,
    loss: boston_house.Loss,
    threshold: float,
):
    "Linear regression with ransac"
    ctx.graph.append("ransac")
    regressor: boston_house.Regressor = "ransac"
    ctx.pivot = "medv" if ctx.pivot is None else ctx.pivot
    model_name = ctx.model.format(regressor=regressor, pivot=ctx.pivot)
    parameters = boston_house.RANSACRegressorParams(
        max_trials=max_trials,
        min_samples=min_samples,
        loss=loss,
        residual_threshold=threshold,
    )
    model = boston_house.RegressionModel(
        regressor=regressor,
        parameters=parameters,
    )
    trainer = boston_house.Trainer(
        model=model,
        pivot=ctx.pivot,
        model_name=model_name,
        test_size=ctx.test_size,
    )
    trainer.train()


@cli.command("decision-tree")
@click.pass_obj
@click.option(
    "--max-depth",
    type=click.INT,
    default=3,
    show_default=True,
    help="Maximal depth of the tree",
)
def decision_tree(ctx: TrainContext, max_depth: int):
    "Decision tree classifier"
    ctx.graph.append("decision-tree")
    regressor: boston_house.Regressor = "forest"
    ctx.pivot = "medv" if ctx.pivot is None else ctx.pivot
    model_name = ctx.model.format(regressor=regressor, pivot=ctx.pivot)
    parameters = boston_house.DecisionTreeParams(
        max_depth=max_depth,
    )
    model = boston_house.RegressionModel(
        regressor=regressor,
        parameters=parameters,
    )
    trainer = boston_house.Trainer(
        model=model,
        pivot=ctx.pivot,
        model_name=model_name,
        test_size=ctx.test_size,
    )
    trainer.train()


@cli.command("random-forest")
@click.pass_obj
@click.option(
    "--n-estimators",
    type=click.INT,
    default=1000,
    show_default=True,
    help="How many trees should be used?",
)
@click.option(
    "--criterion",
    type=click.Choice(CRITERIONS),
    default="squared_error",
    show_choices=True,
    show_default=True,
    help="Which criterion should be used?",
)
@click.option(
    "--n-jobs",
    type=click.IntRange(min=1, max=8),
    default=8,
    show_choices=True,
    show_default=True,
    help="How much cpu - kernels should be used?",
)
def random_forest(
    ctx: TrainContext,
    n_estimators: int,
    criterion: boston_house.RFCriterion,
    n_jobs: boston_house.Kernels,
):
    "Random forest regressor"
    ctx.graph.append("random-forest")
    regressor: boston_house.Regressor = "random"
    ctx.pivot = "medv" if ctx.pivot is None else ctx.pivot
    model_name = ctx.model.format(regressor=regressor, pivot=ctx.pivot)
    parameters = boston_house.RandomForestParams(
        n_estimators=n_estimators,
        criterion=criterion,
        n_jobs=n_jobs,
    )
    model = boston_house.RegressionModel(
        regressor=regressor,
        parameters=parameters,
    )
    trainer = boston_house.Trainer(
        model=model,
        model_name=model_name,
        pivot=ctx.pivot,
        test_size=ctx.test_size,
    )
    trainer.train()


@cli.command()
@click.option(
    "--pivot",
    type=click.Choice(COLUMNS),
    show_choices=True,
    multiple=True,
    required=False,
    help="Pivot to predict",
)
@click.option(
    "--save",
    "-s",
    is_flag=True,
    default=False,
    help="Save picture",
)
@click.option(
    "--hist",
    "-h",
    is_flag=True,
    default=False,
    help="Show or save histogram/s",
)
def stats(pivot: boston_house.Pivot, save: bool, hist: bool):
    "Statistics over Boston house prices"
    explorer = boston_house.Explorer(
        pivot=pivot,
        save=save,
        hist=hist,
    )
    data = explorer.load()
    explorer.plot(data=data)
    explorer.correlation(data=data)


@cli.command("predict")
@click.pass_obj
def predict(ctx: TrainContext):
    "Predict value"
    print(ctx)
    print(f"Model: {ctx.model}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
