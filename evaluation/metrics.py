import numpy as np


def _to_numpy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return y_pred, y_true


def accuracy(y_pred, y_true):
    y_pred, y_true = _to_numpy(y_pred, y_true)
    return float((y_pred == y_true).mean())


def precision(y_pred, y_true, positive_class=1):
    """
    Precision for the positive class.
    Works for 3-class classification (-1, 0, 1).
    """
    y_pred, y_true = _to_numpy(y_pred, y_true)

    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))


def recall(y_pred, y_true, positive_class=1):
    """
    Recall for the positive class.
    """
    y_pred, y_true = _to_numpy(y_pred, y_true)

    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))


def f1_score(y_pred, y_true, positive_class=1):
    """
    F1 score for the positive class.
    """
    p = precision(y_pred, y_true, positive_class)
    r = recall(y_pred, y_true, positive_class)

    if p + r == 0:
        return 0.0

    return float(2 * p * r / (p + r))


def confusion_matrix(y_pred, y_true):
    """
    Returns a 3x3 confusion matrix for classes [-1, 0, 1].
    Rows = true class
    Columns = predicted class
    """
    y_pred, y_true = _to_numpy(y_pred, y_true)

    classes = [-1, 0, 1]
    matrix = np.zeros((3, 3), dtype=int)

    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_c) & (y_pred == pred_c))

    return matrix
