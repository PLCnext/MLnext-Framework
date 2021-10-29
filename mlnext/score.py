""" Module for model evaluation.
"""
import warnings
from typing import Any
from typing import Dict
from typing import List

import numpy as np
from sklearn import metrics


def l2_norm(x: np.array, x_hat: np.array, *, reduce: bool = True) -> np.array:
    """Calculates the l2-norm (euclidean distance) for x and x_hat.
    If reduce is False, then the l2_norm is calculated feature-wise.

    Arguments:
        x (np.array): ground truth.
        x_hat (np.array): prediction.

    Returns:
        np.array: Returns the l2-norm between x and x_hat.

    Example:
        >>> l2_norm(np.array([0.1, 0.2]), np.array([0.14, 0.2]))
        np.array([[0.04]])
        >>> l2_norm(np.array([0.1, 0.2]), np.array([0.14, 0.2]), reduce=False)
        np.array([0.04, 0.0])
    """
    if reduce:
        r = np.sqrt(np.sum((np.array(x) - np.array(x_hat))**2, axis=-1))
        return r.reshape(-1, 1)
    else:
        return np.sqrt((np.array(x) - np.array(x_hat))**2)


def norm_log_likelihood(
        x: np.array,
        mean: np.array,
        log_var: np.array
) -> np.array:
    """Calculates the negative log likelihood that `x` was drawn from
        a normal gaussian distribution defined by `mean` and `log_var`.

    see for reference:
    https://web.stanford.edu/class/archive/cs/cs109/cs109.1192/reader/11%20Parameter%20Estimation.pdf

    ..math::

        Gaussian normal distribution with mean \\mu and variance \\sigma:
        f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp{-\\frac{1}{2}
        (\\frac{x-\\mu}{\\sigma})^2}

        Log likelihood:
        log(p(x | \\mu, \\sigma)) = -0.5 (\\log(2\\pi) + (x-mu)^2/\\sigma^2 +
        \\log(\\sigma^2))

    Args:
        x (np.array): Sample.
        mean (np.array): Mean of the gaussian normal distribution.
        log_var (np.array): Log variance of the gaussian normal distribution.

    Returns:
        np.array: Returns the negative log likelihood.
    """
    a = np.log(2. * np.pi) * np.ones(np.shape(x)) + log_var
    b = (x - mean)**2 / (np.exp(log_var) + 1e-10)

    return 0.5 * (a + b)


def bern_log_likelihood(x: np.array, mean: np.array) -> np.array:
    """Calculates the log likelihood of x being produced by a
        bernoulli distribution parameterized by `mean`.

    see for reference:
    https://web.stanford.edu/class/archive/cs/cs109/cs109.1192/reader/11%20Parameter%20Estimation.pdf

    Args:
        x (np.array): Samples.
        mean (np.array): Mean of the bernoulli distribution.

    Returns:
        np.array: Returns the negative log likelihood.
    """

    a = x * np.log(mean + 1e-10)
    b = (1 - x) * np.log(1 - mean + 1e-10)

    return -(a + b)


def get_threshold(x: np.array, p: float = 100) -> float:
    """Returns the `perc`-th quantile of x.

    Arguments:
        x  (np.array): Input
        p (float): Percentage (0-100).

    Returns:
        float: Returns the threshold at the `perc`-th quantile of x.

    Example:
        >>> get_threshold(np.array([0.0, 1.0]), p=99)
        0.99
    """
    return np.percentile(x, p)


def apply_threshold(x: np.array, *, threshold: float) -> np.array:
    """Applies `threshold t` to `x`. Values greater than the `threshold`
    are 1 and below or equal are 0.

    Arguments:
        x (np.array): Input array.
        t (float): Threshold.

    Returns:
        np.array: Returns the result of the threshold operation.

    Example:
        >>> apply_threshold(np.array([0.1, 0.4, 0.8, 1.0]), threshold=0.5)
        np.array([0, 0, 0, 1, 1])
    """
    return np.where(x > threshold, 1, 0)


def eval_softmax(y: np.array) -> np.array:
    """Turns a multi-class softmax prediction into class labels.

    Arguments:
        y (np.array): Array with softmax probabilites

    Returns:
        np.array: Returns an array of shape (x, 1) with the class labels.

    Example:
        >>> eval_softmax(np.array([[0.1, 0.9], [0.4, 0.6], [0.7, 0.3]]))
        np.array([[1], [1], [0]])
    """
    return np.argmax(y, axis=-1).reshape(-1, 1)


def eval_sigmoid(y: np.array, *, invert: bool = False) -> np.array:
    """Turns a binary-class sigmoid prediction into 0-1 class labels.

    Args:
        y (np.array): Array with sigmoid probabilities
        invert (bool): Whether to invert the labels. (0->1, 1->0)

    Returns:
        np.array: Returns the binary class labels.

    Example:
        >>> eval_sigmoid(y=np.array([0.1, 0.6, 0.8, 0.2]))
        np.array([[0],[1],[1],[0]])
    """
    y = (y > 0.5) * 1.
    if not invert:
        return y.reshape(-1, 1)
    else:
        return (1. - y).reshape(-1, 1)


def moving_average(x: np.array, step: int = 10, mode='full') -> np.array:
    """Calculates the moving average for X with stepsize `step`.

    Args:
        X (np.array): 1-dimensional array.
        step (int, optional): Stepsize. Defaults to 10.
        mode (str, optional): Mode, see np.convolve. Defaults to 'full'.

    Returns:
        np.array: Returns the moving average.

    Example:
        >>> moving_average(np.array([1, 2, 3, 4]), step=2)
        np.array([0.5, 1.5, 2.5, 3.5, 2.])
    """
    return np.convolve(x, np.ones((step,)) / step, mode=mode)


def eval_metrics(y: np.array, y_hat: np.array) -> Dict[str, Any]:
    """Calculates accuracy, f1, precision, recall and AUC scores.

    Arguments:
        y (np.array): Ground truth.
        y_hat (np.array): Predictions.

    Returns:
        Dict[str, Any]: Returns a dict with all scores.

    Example:
        >>> y, y_hat = np.ones((10, 1)), np.ones((10, 1))
        >>> eval_metrics(y, y_hat)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    scores = {
        'accuracy': metrics.accuracy_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
        'f1': metrics.f1_score,
        'AUC': metrics.roc_auc_score
    }

    if y.shape != y_hat.shape:
        warnings.warn(f'Shapes unaligned {y.shape} and {y_hat.shape}.')

    length = min(y.shape[0], y_hat.shape[0])
    results = {}
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
        y_hat (np.array): Prediction.

    Returns:
        Dict[str, Any]: Returns a dict with all scores.

    Example:
        >>> y = [np.ones((10, 1)), np.zeros((10, 1))]
        >>> y_hat = [np.ones((10, 1)), np.zeros((10, 1))]
        >>> eval_metrics_all(y, y_hat)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    y_ = []
    y_hat_ = []
    for (x, xx) in zip(y, y_hat):
        if x.shape != xx.shape:
            warnings.warn(f'Shapes unaligned {x.shape} and {xx.shape}.')

        length = min(x.shape[0], xx.shape[0])
        y_.append(x[:length])
        y_hat_.append(xx[:length])

    y_ = np.vstack(y_)
    y_hat_ = np.vstack(y_hat_)

    return eval_metrics(y_, y_hat_)
