"Ensemble learning"
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import numpy as np
# pylint: disable=invalid-name


class MajorityVotesClassifier(BaseEstimator, ClassifierMixin):
    """Majority Votes Classifier as ensemble

    Parameters
    ----------
    classifiers : array-like, shape=[n_classifiers]
        different classifiers of the ensemble

    vote : str, {'classlabel', 'probability'}
        default: 'classlabel'

    weights: array-like, shape = [n_classifiers]
        Optional, default None
    """

    def __init__(self, classifiers, vote="classlabel", weights=None):
        "constructor"
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        self.lablenc_ = LabelEncoder()
        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X, y):
        """Adjust classificators
        """
        if self.vote not in ("probability", "classlabel"):
            raise ValueError(
                f"Vote must be probability or classlabel not {self.vote}",
            )
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(
                "weights and classifiers must have the same length",
            )
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        "Predict class label of x_data"
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions,
            )

        maj_vote = self.lablenc_.inverse_transform(maj_vote)

        return maj_vote

    def get_params(self, deep=True):
        "Classifier param names for gridsearch"
        if not deep:
            return super(MajorityVotesClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value
            return out

    def predict_proba(self, X):
        "Predict class probabilities"
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)

        return avg_proba
