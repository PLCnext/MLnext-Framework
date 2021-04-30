from typing import Any
from typing import Dict
from typing import List

import numpy as np
from scipy import stats
from sklearn import metrics


def l2_norm(x: np.array, x_hat: np.array) -> np.array:
    """Calculates the l2-norm (euclidean distance) for x and x_hat.

    Arguments:
        x (np.array): ground truth.
        x_hat (np.array): prediction.

    Returns:
        np.array: Returns the l2-norm between x and x_hat.
    """
    r = np.sqrt(np.sum((np.array(x) - np.array(x_hat))**2, axis=-1))
    return r.reshape(-1, 1)


def get_threshold(x: np.array, p: float = 100) -> float:
    """Returns the `perc`% max.

    Arguments:
        x  (np.array): Input
        p (float): Percentage (0-100).

    Returns:
        float: Returns the threshold at `perc`% max.
    """
    return stats.scoreatpercentile(x, p)


def apply_threshold(x: np.array, *, threshold: float) -> np.array:
    """Applies `threshold t` to `x`. Values greater than the `threshold`
    are 1 and below or equal are 0.

    Arguments:
        x (np.array): Input array.
        t (float): Threshold.

    Returns:
        np.array: Returns the result of the threshold operation.
    """
    return np.where(x > threshold, 1, 0)


def eval_softmax(y: np.array) -> np.array:
    """Turns a multi-class softmax prediction into class labels.

    Arguments:
        y (np.array): Array with softmax probabilites

    Returns:
        np.array: Returns an array of shape (x, 1) with the class labels.
    """
    return np.argmax(y, axis=-1).reshape(-1, 1)


def eval_sigmoid(*, y: np.array, invert: bool = False) -> np.array:
    """Turns a binary-class sigmoid prediction into 0-1 class labels.

    Args:
        y (np.array): Array with sigmoid probabilities
        invert (bool): Whether to invert the labels. (0->1, 1->0)

    Returns:
        np.array: Returns the binary class labels.
    """
    y = (y > 0.5) * 1.
    if not invert:
        return y.reshape(-1, 1)
    else:
        return (1. - y).reshape(-1, 1)


def moving_average(X: np.array, step: int = 10, mode='full') -> np.array:
    """Calculates the moving average for X with stepsize `step`.

    Args:
        X (np.array): 1-dimensional array.
        step (int, optional): Stepsize. Defaults to 10.
        mode (str, optional): Mode, see np.convolve. Defaults to 'full'.

    Returns:
        np.array: Returns the moving average.
    """
    return np.convolve(X, np.ones((step,)) / step, mode=mode)


def eval_metrics(y: np.array, y_hat: np.array) -> Dict[str, Any]:
    """Calculates accuracy, f1, precision, recall and AUC scores.

    Arguments:
        y (np.array): Ground truth.
        y_hat (np.array): Predictions.

    Returns:
        Dict[str, Any]: Returns a dict with all scores.
    """
    scores = {
        'accuracy': metrics.accuracy_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
        'f1': metrics.f1_score,
        'AUC': metrics.roc_auc_score
    }

    results = {}
    length = min(len(y), len(y_hat))
    try:
        for key in scores:
            results[key] = scores[key](y[:length], y_hat[:length])
    except Exception:
        pass
    finally:
        return results


def eval_metrics_all(y: List[np.array],
                     y_hat: List[np.array]) -> Dict[str, Any]:
    """Calculates combined accuracy, f1, precision, recall and AUC scores for
    multiple arrays. The arrays are shorted to the minimum length of the
    corresponding partner and stacked on top of each other to calculated the
    combined scores.

    Arguments:
        y (np.array): Ground truth.
        y_hat {array-like} -- Prediction

    Returns:
        Dict[str, Any]: Returns a dict with all scores.
    """
    y_ = []
    y_hat_ = []

    for (x, xx) in zip(y, y_hat):
        length = min(len(x), len(xx))
        y_.append(x[:length])
        y_hat_.append(xx[:length])

    y_ = np.vstack(y_)
    y_hat_ = np.vstack(y_hat_)

    return eval_metrics(y_, y_hat_)
