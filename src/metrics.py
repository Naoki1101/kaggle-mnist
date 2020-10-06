import numpy as np


def accuracy(y_true: np.array, y_pred: np.array) -> float:
    y_pred_class = np.argmax(y_pred, axis=1)

    acc = (y_true == y_pred_class).sum() / len(y_true)
    return acc
