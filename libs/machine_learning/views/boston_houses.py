"""Loader for Boston house data"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class BostonHouses:
    "Boston houses"

    data: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        "Load boston house data from parquet file"
        self.data = pd.read_parquet(
            "data/boston_house.parquet",
            engine="pyarrow",
        )

    def split(
        self,
        test_size: float = 0.2,
        stratify: bool = False,
        random_state: int = 1,
        pivot: str = "MEDV",
    ) -> TrainTest:
        "Split boston house in train and test data"
        columns = [col for col in self.data.columns if col != pivot]
        return train_test_split(
            self.data[columns].values,
            self.data[pivot].values,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data[pivot] if stratify else None,
        )
