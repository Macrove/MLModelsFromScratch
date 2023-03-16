import numpy as np

def accuracy(y_pred, y):
    return np.sum(y_pred == y)/len(y)