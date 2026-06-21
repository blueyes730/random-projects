import numpy as np

from sklearn import datasets

class LogisticRegression:
    def __init__(self, eta=1, shuffle=False, num_epochs=10, random_state=1):
        self.eta = eta
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.w_initialized = False
        self.shuffle = shuffle
    
    def fit(self, X, y):
        """ Fit the perceptron on the training data. 

        Parameters:
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]

        Returns:
        self : object
        """

        self.losses = []
        self.initialize_weights(X.shape[1])
        for _ in range(self.num_epochs):
            if self.shuffle:
                X, y = self.shuffle_dataset(X, y)
            losses = []
            errors = 0
            for xi, yi in zip(X, y):
                losses.append(self.update(xi, yi))
            avg_loss = np.mean(losses)
            self.losses.append(avg_loss)
        return self
    
    def continue_fit(self, X, y):
        if not self.w_initialized:
            self.initialize_weights(X.shape[1]) # X.shape[1] is num_features
        
        if y.ravel().shape[0] > 1:
            # y has multiple samples, we need to assume it is a batch and process as such
            for xi, yi in zip(X, y):
                self.update(xi, yi)
        else:
            # Single sample was passed in
            self.update(X,y)
        return self

    def initialize_weights(self, m):
        # x dot w has to work, x is shape [n_samples, n_features], so w is [n_features]
        self.w = np.random.RandomState(self.random_state).normal(0, 0.01, size=m)
        self.b = 0.0
        self.w_initialized = True

    def net_input(self, X):
        return np.dot(X, self.w) + self.b
    
    def update(self, xi, yi):
        output = self.predict(xi)
        error = yi - output
        self.w += self.eta * error * xi
        self.b += self.eta * error
        loss = (-yi*np.log(output)) - ((1.0-yi) * np.log(1.0 - output))
        return loss
        
    def activation(self, z):
        return 1. /(1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
    def shuffle_dataset(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]   
    
if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    print(X.shape)
    
    ppn = LogisticRegression(eta=0.1, num_epochs=10)
    ppn.fit(X, y)
