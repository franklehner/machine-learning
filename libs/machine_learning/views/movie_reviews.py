"Prepare movie revie data"
import multiprocessing
import re
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

tag_regex = re.compile(r"<[^>]*>")
emo_regex = re.compile(r"(?::|;|=)(?:-)?(?:/)|(|D|P)")
regex = re.compile(r"[\W]+")
Tensor = np.ndarray
TrainTest = Tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class Reviews:
    "Movie reviews"

    data: pd.DataFrame = field(init=False)

    def __post_init__(self):
        start = time.perf_counter()
        cpu_count = multiprocessing.cpu_count()
        self.load()
        reviews = np.split(self.data, cpu_count)

        with multiprocessing.Pool(processes=cpu_count) as pool:
            cleaned = pool.map(self.preprocessor, reviews)

        end = time.perf_counter() - start
        self.data = pd.concat(cleaned, ignore_index=True)
        print(f"Elapsed time: {end}")

    def load(self) -> None:
        "load Parquet file"
        self.data = pd.read_parquet("data/movie_data.parquet", engine="pyarrow")

    def preprocessor(self, df: pd.DataFrame) -> pd.DataFrame:
        "Remove emoticons and tags"
        df["reviews"] = df["reviews"].str.replace(tag_regex.pattern, "", regex=True)
        df["reviews"] = df["reviews"].str.replace(emo_regex.pattern, "", regex=True)

        return df

    def split(
        self,
        test_size: float = 0.5,
        stratify: bool = False,
        random_state: int = 1,
    ) -> TrainTest:
        "split data"
        return train_test_split(
            self.data["reviews"].values,
            self.data["sentiment"].values,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data["sentiment"] if stratify else None,
        )
