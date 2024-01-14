"Vectorizer for movie classifier"
from sklearn.feature_extraction.text import HashingVectorizer

from libs.machine_learning.views import prepare_dataset


def get_vectorizer() -> HashingVectorizer:
    "get the vectorizer"
    vect = HashingVectorizer(
        decode_error="ignore",
        n_features=2**21,
        preprocessor=None,
        tokenizer=prepare_dataset.tokenizer,
    )
    return vect
