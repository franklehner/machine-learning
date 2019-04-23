import pandas as _pd
import numpy as _np

import libs.domain.usecases.perceptron as _usecase


def prepare_data(path):
    return


def predict_iris(data):
    return


def plot(plot_type, x_range, y_range, legend, label):
    pass


def run(path):
    data = prepare_data(path)

    labels = data.iloc[:100, 4].values
    labels = _np.where(labels == "Iris-setosa", -1, 1)
    train_data = data.iloc[:100, [0, 2]].values
