"""Iris Data loader
"""
import dataclasses
import pandas as pd
import numpy as np


@dataclasses.dataclass
class Iris:
    """Iris data
    """
    data: np.ndarray = dataclasses.field(init=False)
    targets: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.load()

    def _load_iris(self) -> pd.DataFrame:
        """Load iris data
        """
        return pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            header=None,
        )

    def load(self):
        """Load
        """
        iris_data = self._load_iris()
        self.targets = iris_data.iloc[0:, 4].values
        self.data = iris_data.iloc[0:, 0:4].values
