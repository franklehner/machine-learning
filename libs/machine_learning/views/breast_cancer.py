"Loader for breast cancer data"
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_FILENAME = "data/breast_cancer_wisconsin.csv"
Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class BreastCancer:
    "breast cancer data of wisconsin"

    data: Tensor = field(init=False)
    targets: Tensor = field(init=False)
    feature_names: List[str] = field(init=False, default_factory=list)
    target_names: Tensor = field(init=False)

    def __post_init__(self):
        self.load()

    def load(self) -> None:
        "load breast cancer data"
        df = pd.read_csv(DEFAULT_FILENAME)
        del df["ID"]
        value_cols = [col for col in df.columns if col != "diagnosis"]
        self.data = df[value_cols].values
        targets = df["diagnosis"].values
        le = LabelEncoder()
        self.targets = le.fit_transform(targets)
        self.target_names = le.classes_
        self.feature_names = value_cols

    def split(
        self,
        test_size: Optional[float] = 0.3,
        stratify: bool = False,
        random_state: Optional[int] = None,
    ) -> TrainTest:
        "Split data"
        if stratify:
            return train_test_split(
                self.data,
                self.targets,
                test_size=test_size,
                random_state=random_state,
                stratify=self.targets,
            )
        return train_test_split(
            self.data,
            self.targets,
            test_size=test_size,
            random_state=random_state,
        )
