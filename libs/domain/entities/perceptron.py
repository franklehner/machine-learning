import abc as _abc


class Plotter:
    pass


class AbstractUrlReaderPandas(_abc.ABC):
    """
    Base class for Panda file streams
    """

    @_abc.abstractmethod
    def read(self, url):
        raise NotImplementedError

    @_abc.abstractmethod
    def plot(self, x_range, y_range, legend, label):
        raise NotImplementedError
