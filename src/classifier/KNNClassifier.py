import numpy as np

class KNNClassifier:
    
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def __euclid_dist(self, x):
        return np.array([np.sum(np.power(x1-x, 2)) for x1 in self.X])

    def predict(self, X):
        y_preds = np.zeros(X.shape[0], dtype=np.uint8)
        for idx, x in enumerate(X):
            dist = self.__euclid_dist(x)
            nearest_k = np.take_along_axis(self.y, np.argsort(dist, axis=0), axis=0)[:self.k]
            labels, counts = np.unique(nearest_k, return_counts=True)
            y_preds[idx] = labels[np.argmax(counts)]

        return np.array(y_preds)