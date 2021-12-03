""" Module for model evaluation.
"""
import warnings
from typing import Dict
from typing import List

import numpy as np
from sklearn import metrics

from mlnext.utils import check_ndim
from mlnext.utils import check_size
from mlnext.utils import truncate


def l2_norm(
    x: np.ndarray,
    x_hat: np.ndarray,
    *,
    reduce: bool = True
) -> np.ndarray:
    """Calculates the l2-norm (euclidean distance) for x and x_hat.
    If reduce is False, then the l2_norm is calculated feature-wise.

    Arguments:
        x (np.ndarray): ground truth.
        x_hat (np.ndarray): prediction.

    Returns:
        np.ndarray: Returns the l2-norm between x and x_hat.

    Example:
        >>> l2_norm(np.array([0.1, 0.2]), np.array([0.14, 0.2]))
        np.ndarray([[0.04]])
        >>> l2_norm(np.array([0.1, 0.2]), np.array([0.14, 0.2]), reduce=False)
        np.ndarray([0.04, 0.0])
    """
    if reduce:
        r = np.sqrt(np.sum((np.array(x) - np.array(x_hat))**2, axis=-1))
        return r.reshape(-1, 1)
    else:
        return np.sqrt((np.array(x) - np.array(x_hat))**2)


def norm_log_likelihood(
        x: np.ndarray,
        mean: np.ndarray,
        log_var: np.ndarray
) -> np.ndarray:
    """Calculates the negative log likelihood that `x` was drawn from a normal
    gaussian distribution defined by `mean` and `log_var`.

    .. math::

        f(x|\\mu, \\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp{-\\frac{1}
        {2}(\\frac{x-\\mu}{\\sigma})^2}

        \\text{Log likelihood}:
        log(f(x | \\mu, \\sigma)) = -0.5 (\\log(2\\pi) + (x-\\mu)^2/\\sigma^2 +
        \\log(\\sigma^2))

    Args:
        x (np.ndarray): Sample.
        mean (np.ndarray): Mean of the gaussian normal distribution.
        log_var (np.ndarray): Log variance of the gaussian normal distribution.

    Returns:
        np.ndarray: Returns the negative log likelihood.
    """
    a = np.log(2. * np.pi) * np.ones(np.shape(x)) + log_var
    b = (x - mean)**2 / (np.exp(log_var) + 1e-10)

    return 0.5 * (a + b)


def bern_log_likelihood(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Calculates the log likelihood of x being produced by a bernoulli
    distribution parameterized by `mean`.

    .. math::

        LL(x|\\text{mean}) = x \\cdot \\log(\\text{mean}) +
        (1 - x) \\cdot \\log(\\text{mean})

    Args:
        x (np.ndarray): Samples.
        mean (np.ndarray): Mean of the bernoulli distribution.

    Returns:
        np.ndarray: Returns the negative log likelihood.
    """

    a = x * np.log(mean + 1e-10)
    b = (1 - x) * np.log(1 - mean + 1e-10)

    return -(a + b)


def kl_divergence(
    mean: np.ndarray,
    log_var: np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 1.0
) -> np.ndarray:
    """Calculates the kl divergence kld(q||p) between a normal gaussian `p`
    (prior_mean, prior_std) and a normal distribution `q` parameterized
    by `mean` and `log_var`.

    Args:
        mean (np.ndarray): Mean of q.
        log_var (np.ndarray): Log variance of q.
        prior_mean (float): Mean of the prior p. Defaults to 0.0.
        prior_std (float): Standard deviation of the prior p. Defaults to 1.0.

    Returns:
        np.ndarray: Returns the kl divergence between two normal distributions.
    """

    prior_mean = np.ones(mean.shape) * prior_mean  # type:ignore
    prior_std = np.ones(log_var.shape) * prior_std  # type:ignore

    # see https://stats.stackexchange.com/a/7443
    # log(o_2 / o_1)
    a = 0.5 * log_var - np.log(prior_std)
    # o_1^2 + (mu_1 - mu_2)^2
    b = np.square(prior_std) + np.square(prior_mean - mean)
    # 2o_2^2
    c = 2 * np.exp(log_var)

    return a + (b / c) - 0.5


def get_threshold(x: np.ndarray, p: float = 100) -> float:
    """Returns the `perc`-th percentile of x.

    Arguments:
        x  (np.ndarray): Input
        p (float): Percentage (0-100).

    Returns:
        float: Returns the threshold at the `perc`-th percentile of x.

    Example:
        >>> get_threshold(np.ndarray([0.0, 1.0]), p=99)
        0.99
    """
    return np.percentile(x, p)


def apply_threshold(
    x: np.ndarray,
    *,
    threshold: float,
    pos_label: int = 1,
    neg_label: int = 0,
) -> np.ndarray:
    """Applies `threshold t` to `x`. Values that are greater than or equal
    than the `threshold` are changed to `pos_label` and below to `neg_label`.

    Arguments:
        x (np.ndarray): Input array.
        t (float): Threshold.
        pos_label (int): Label for the class above the threshold. Defaults to
          1.
        neg_label (int): Label fot the class below the threshold. Default to
          0.

    Returns:
        np.ndarray: Returns the result of the threshold operation.

    Example:
        >>> apply_threshold(np.array([0.1, 0.4, 0.8, 1.0]), threshold=0.5)
        np.ndarray([0, 0, 0, 1, 1])
    """
    return np.where(x >= threshold, pos_label, neg_label)


def eval_softmax(y: np.ndarray) -> np.ndarray:
    """Turns a multi-class softmax prediction into class labels.

    Arguments:
        y (np.ndarray): Array with softmax probabilites

    Returns:
        np.ndarray: Returns an array of shape (x, 1) with the class labels.

    Example:
        >>> eval_softmax(np.array([[0.1, 0.9], [0.4, 0.6], [0.7, 0.3]]))
        np.ndarray([[1], [1], [0]])
    """
    return np.argmax(y, axis=-1).reshape(-1, 1)


def eval_sigmoid(y: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """Turns a binary-class sigmoid prediction into 0-1 class labels.

    Args:
        y (np.ndarray): Array with sigmoid probabilities
        invert (bool): Whether to invert the labels. (0->1, 1->0)

    Returns:
        np.ndarray: Returns the binary class labels.

    Example:
        >>> eval_sigmoid(y=np.array([0.1, 0.6, 0.8, 0.2]))
        np.ndarray([[0],[1],[1],[0]])
    """
    y = (y > 0.5) * 1.
    if not invert:
        return y.reshape(-1, 1)
    else:
        return (1. - y).reshape(-1, 1)


def moving_average(x: np.ndarray, step: int = 10, mode='full') -> np.ndarray:
    """Calculates the moving average for X with stepsize `step`.

    Args:
        X (np.ndarray): 1-dimensional array.
        step (int, optional): Stepsize. Defaults to 10.
        mode (str, optional): Mode, see np.convolve. Defaults to 'full'.

    Returns:
        np.ndarray: Returns the moving average.

    Example:
        >>> moving_average(np.array([1, 2, 3, 4]), step=2)
        np.ndarray([0.5, 1.5, 2.5, 3.5, 2.])
    """
    return np.convolve(x, np.ones((step,)) / step, mode=mode)


def eval_metrics(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    """Calculates accuracy, f1, precision, recall and AUC scores.

    Arguments:
        y (np.ndarray): Ground truth.
        y_hat (np.ndarray): Predictions.

    Returns:
        Dict[str, float]: Returns a dict with all scores.

    Example:
        >>> y, y_hat = np.ones((10, 1)), np.ones((10, 1))
        >>> eval_metrics(y, y_hat)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
        ... 'auc': 1.0}
    """
    scores = {
        'accuracy': metrics.accuracy_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
        'f1': metrics.f1_score,
        'auc': metrics.roc_auc_score
    }

    if y.shape != y_hat.shape:
        warnings.warn(f'Shapes unaligned {y.shape} and {y_hat.shape}.')

    (y, y_hat), = truncate((y, y_hat))
    results = {}
    try:
        for key in scores:
            results[key] = scores[key](y, y_hat)
    except Exception:
        pass
    finally:
        return results


def eval_metrics_all(
    y: List[np.ndarray],
    y_hat: List[np.ndarray]
) -> Dict[str, float]:
    """Calculates combined accuracy, f1, precision, recall and AUC scores for
    multiple arrays. The arrays are shorted to the minimum length of the
    corresponding partner and stacked on top of each other to calculated the
    combined scores.

    Arguments:
        y (np.ndarray): Ground truth.
        y_hat (np.ndarray): Prediction.

    Returns:
        Dict[str, float]: Returns a dict with all scores.

    Example:
        >>> y = [np.ones((10, 1)), np.zeros((10, 1))]
        >>> y_hat = [np.ones((10, 1)), np.zeros((10, 1))]
        >>> eval_metrics_all(y, y_hat)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
        ... 'auc': 1.0}
    """
    if len(y) != len(y_hat):
        raise ValueError('y and y_hat must have the same number elements.')

    # allow 1d or 2d arrays with the 2nd dimension of 1
    check_ndim(*y, *y_hat, ndim=2, strict=False)
    check_size(*y, *y_hat, size=1, axis=1, strict=False)

    y = list(map(lambda x: x.reshape(-1), y))
    y_hat = list(map(lambda x: x.reshape(-1), y_hat))

    # truncate corresponding arrays to the same length
    y_, y_hat_ = np.hstack(list(truncate(*zip(y, y_hat))))

    return eval_metrics(y_, y_hat_)
