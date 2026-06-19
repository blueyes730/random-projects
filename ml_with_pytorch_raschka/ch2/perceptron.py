import numpy as np

from sklearn import datasets

class Perceptron:
    def __init__(self, eta=1, num_epochs=10, random_state=1):
        self.eta = eta
        self.num_epochs = num_epochs
        self.random_state = random_state

        self.errors = []
    
    def fit(self, X, y):
        """ Fit the perceptron on the training data. 

        Parameters:
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]

        Returns:
        self : object
        """

        # x dot w has to work, x is shape [n_samples, n_features], so w is [n_features]
        self.w = np.random.RandomState(self.random_state).normal(0, 0.01, size=X.shape[1])
        self.b = 0.0

        for _ in range(self.num_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    print(X.shape)
    
    ppn = Perceptron(eta=0.1, num_epochs=10)
    ppn.fit(X, y)
