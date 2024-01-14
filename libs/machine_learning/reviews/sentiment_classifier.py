"Sentiment analysis for movies"
from dataclasses import dataclass
from typing import Dict, List

import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from libs.machine_learning.views import prepare_dataset

Tensor = np.ndarray


@dataclass
class SentimentAnalysis:
    "sentiment analysis class"

    path: str
    stem: bool

    def __str__(self):
        text = f"Path: {self.path}, use stemmer: {self.stem}"
        return text

    def get_data(self) -> prepare_dataset.Data:
        "get the movie data"
        dataset = prepare_dataset.DataSet(name="movie")

        return dataset.get_dataset()

    def make_pipeline(self) -> Pipeline:
        "make the pipeline"
        tfidf = TfidfVectorizer(
            strip_accents=None,
            lowercase=False,
            preprocessor=None,
        )
        lr_tfidf = Pipeline(
            [
                ("vect", tfidf),
                ("clf", LogisticRegression(random_state=0, solver="liblinear")),
            ],
        )

        return lr_tfidf

    def get_grid_params(self) -> List[Dict]:
        "get the params of the grid"
        nltk.download("stopwords")
        stop = stopwords.words("english")
        param_grid = [
            {
                "vect__ngram_range": [(1, 1)],
                "vect__stop_words": [stop, None],
                "vect__tokenizer": [
                    None,
                    prepare_dataset.tokenizer,
                    prepare_dataset.tokenizer_porter,
                ],
                "clf__penalty": ["l1", "l2"],
                "clf__C": [1.0, 10.0, 100.0],
            },
            {
                "vect__ngram_range": [(1, 1)],
                "vect__stop_words": [stop, None],
                "vect__tokenizer": [
                    None,
                    prepare_dataset.tokenizer,
                    prepare_dataset.tokenizer_porter,
                ],
                "vect__use_idf": [False],
                "vect__norm": [None],
                "clf__penalty": ["l1", "l2"],
                "clf__C": [1.0, 10.0, 100.0],
            },
        ]

        return param_grid

    def train(self) -> None:
        "Train text"
        data = self.get_data()
        x_train, x_test, y_train, y_test = data.split(
            test_size=0.5,
            random_state=1,
        )
        pipeline = self.make_pipeline()
        grid_params = self.get_grid_params()
        gs_lr_tfidf = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_params,
            scoring="accuracy",
            cv=5,
            verbose=1,
            n_jobs=-1,
        )
        gs_lr_tfidf.fit(x_train, y_train)
        print(
            f"Best parameters:\n{gs_lr_tfidf.best_params_}",
        )
        score = gs_lr_tfidf.score(x_test, y_test)
        print(f"Test score: {score}")
