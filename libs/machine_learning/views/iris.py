"""Iris Data loader
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]
Name = Literal["petal", "sepal"]
Dim = Literal["length", "width"]


@dataclass
class Iris:
    """Iris data"""

    data: Tensor = field(init=False)
    targets: Tensor = field(init=False)
    feature_names: List[str] = field(init=False, default_factory=list)
    target_names: Tensor = field(init=False)

    def __post_init__(self):
        self.load()

    def load(self) -> None:
        """Load"""
        iris_data = load_iris()
        self.data = iris_data["data"]
        targets = iris_data["target"]
        le = LabelEncoder()
        self.targets = le.fit_transform(targets)
        self.feature_names = iris_data["feature_names"]
        self.target_names = iris_data["target_names"]

    def split(
        self,
        count: Optional[int] = None,
        test_size: Optional[float] = 0.3,
        random_state: Optional[int] = 1,
        stratify: bool = False,
    ) -> TrainTest:
        "Split data into train and test data"
        if count is not None:
            data = self.data[:count]
            targets = self.targets[:count]
        else:
            data = self.data
            targets = self.targets

        if stratify:
            return train_test_split(
                data,
                targets,
                test_size=test_size,
                random_state=random_state,
                stratify=targets,
            )
        return train_test_split(
            data,
            targets,
            test_size=test_size,
            random_state=random_state,
        )


    def standardize(
        self, data_train: Tensor, data_test: Tensor
    ) -> Tuple[Tensor, Tensor]:
        "standardize"
        sc = StandardScaler()
        sc.fit(data_train)
        train = sc.transform(data_train)
        test = sc.transform(data_test)

        return train, test

    def name2idx(self, name: Name, dim: Optional[Dim] = None) -> Union[int, List[int]]:
        "Map features from name to idx"
        features = [tuple(row.split()[:2]) for row in self.feature_names]
        indexes: List[int] = []
        if dim is None:
            for idx, row in enumerate(features):
                if row[0] == name:
                    indexes.append(idx)
            return indexes

        return features.index((name, dim))

    def idx2name(self, idx: List[int]) -> List[str]:
        "Map feature names to index"
        names: List[str] = []
        for i in idx:
            names.append(self.feature_names[i])

        return names
