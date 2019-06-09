import pandas as _pd
import libs.domain.entities.perceptron as _entity


class UrlRepository(AbstractUrlReaderPandas):
    """
    Handle data from url
    """

    @classmethod
    def read(cls, url):
        return _pd.read_csv(url, sep=",")

    @classmethod
    def plot(cls, x_range, y_range, legend, label):
        pass
