import pandas as _pd
import libs.domain.entities.perceptron as _entity


class UrlRepository(AbstractUrlReaderPandas):
    """
    Handle data from url
    """

    @classmethod
    def read(self, url):
        return _pd.read_csv(url, sep=",")
