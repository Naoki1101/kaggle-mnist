import numpy as np


def accuracy(y_true: np.array, y_pred: np.array) -> float:
    acc = (y_true == y_pred).sum() / len(y_true)
    return acc
