"""Hierarichal clustering
"""
from dataclasses import dataclass, field
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN

Tensor = np.ndarray
Method = Literal[
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
]
Metric = Literal["euclidean"]


@dataclass
class ScipyCluster:
    "square form of data"

    data: pd.DataFrame
    metric: Metric = "euclidean"
    method: Method = "complete"
    labels: List = field(init=False, default_factory=list)

    def __post_init__(self):
        columns = self.data.columns
        self.labels = [f"ID_{idx}" for idx in range(len(self.data))]
        self.data = pd.DataFrame(self.data.values, columns=columns, index=self.labels)

    def get_squareform(self) -> pd.DataFrame:
        "get the squareform of the data"
        return pd.DataFrame(
            squareform(pdist(self.data, metric=self.metric)),
            columns=self.labels,
            index=self.labels,
        )

    def get_clusters(self, show: bool = True) -> pd.DataFrame:
        "get the clusters of the data"
        row_clusters = linkage(
            self.data.values,
            method=self.method,
            metric=self.metric,
        )
        if show:
            self.draw(row_clusters=row_clusters)
        clusters = pd.DataFrame(
            row_clusters,
            columns=["row_1", "row_2", "distance", "objects_in_cluster"],
            index=[f"cluster_{i + 1}" for i in range(row_clusters.shape[0])],
        )
        clusters["row_1"] = clusters["row_1"].map(int)
        clusters["row_2"] = clusters["row_2"].map(int)
        clusters["objects_in_cluster"] = clusters["objects_in_cluster"].map(int)

        return clusters

    def draw(self, row_clusters: Tensor) -> None:
        "draw dendogram"
        _ = dendrogram(
            row_clusters,
            labels=self.labels,
        )
        plt.tight_layout()
        plt.ylabel("Euclidean distance")
        plt.show()


@dataclass
class Agglomerative:
    "Aglomerative clustering"

    n_clusters: int = 3
    metric: Metric = "euclidean"
    linkage: Method = "complete"

    def build(self) -> AgglomerativeClustering:
        "Build cluster"
        return AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage,
        )


@dataclass
class DBSCluster:
    "DBSCAN"

    eps: float = 0.2
    min_samples: int = 5
    metric: Metric = "euclidean"

    def build(self) -> DBSCAN:
        "build cluster"
        return DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
