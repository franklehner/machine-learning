"Wine data loader"
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_FILENAME = "data/wine.csv"
Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class Wine:
    "Wine data"

    data: Tensor = field(init=False)
    targets: Tensor = field(init=False)
    feature_names: List[str] = field(init=False, default_factory=list)
    target_names: Tensor = field(init=False)

    def __post_init__(self):
        self.load()

    def load(self) -> None:
        "load data"
        df = pd.read_csv(DEFAULT_FILENAME)
        value_cols = [col for col in df.columns if col != "Klassenbezeichnung"]
        self.data = df[value_cols].values
        le = LabelEncoder()
        self.targets = le.fit_transform(np.array(df["Klassenbezeichnung"].values))
        self.feature_names = value_cols
        self.target_names = df["Klassenbezeichnung"].unique()

    def split(self, count: Optional[int] = None) -> TrainTest:
        "Split data into train and test data"
        if count is not None:
            data = self.data[:count]
            targets = self.targets[:count]
        else:
            data = self.data
            targets = self.targets

        return train_test_split(data, targets, stratify=targets)

    def standardize(self, x_train: Tensor, x_test: Tensor) -> Tuple[Tensor, Tensor]:
        "Standardize data"
        sc = StandardScaler()
        sc.fit(x_train)
        train = sc.transform(x_train)
        test = sc.transform(x_test)

        return train, test
