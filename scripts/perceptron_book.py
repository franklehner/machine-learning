import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt


class Perceptron:
    """
    Perceptron-Classifier

    eta: Float
        learning rate (between 0.0 and 1.0)
    n_iter: int
        number of epochs
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = _np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return _np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return _np.where(self.net_input(X) >= 0.0, 1, -1)


df = _pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None
)
print("iris data loaded...")
y = df.iloc[:100, 4].values
y = _np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[:100, [0, 2]].values
_plt.scatter(
    X[:50, 0],
    X[:50, 1],
    color="red",
    marker="o",
    label="setosa"
)
_plt.scatter(
    X[50:100, 0],
    X[50:100, 1],
    color="blue",
    marker="x",
    label="versicolor"
)
_plt.xlabel("Länge des Kelchblatts [cm]")
_plt.ylabel("Länge des Blütenblatts [cm]")
_plt.legend(loc="upper left")
_plt.show()

ppn = Perceptron(eta=0.1, n_iter=6)
ppn.fit(X, y)
_plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker="o")
_plt.xlabel("Epochen")
_plt.ylabel("Anzahl der Updates")
_plt.show()
