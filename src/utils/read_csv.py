import numpy as np
import pandas as pd

def read_csv(path):
    data = pd.read_csv(path)
    X = np.stack((np.array(data['x']), np.array(data['y'])), axis = 1)
    y = np.array(data['label'], dtype=np.uint8)
    return X, y