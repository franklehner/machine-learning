import abc as _abc


class AbstractUrlReaderPandas(_abc.ABC):
    """
    Base class for Panda file streams
    """

    @_abc.abstractmethod
    def read(self, url):
        raise NotImplementedError
