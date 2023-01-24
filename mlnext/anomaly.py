"""Module for analyzing anomalies.
"""
import typing as T
import warnings
from operator import ge
from operator import gt

import numpy as np
import pandas as pd

from .data import detemporalize
from .utils import check_ndim
from .utils import check_shape
from .utils import truncate

__all__ = [
    'find_anomalies',
    'recall_anomalies',
    'rank_features',
    'apply_point_adjust',
    'apply_point_adjust_score',
]


def find_anomalies(
    y: T.Union[np.ndarray, pd.DataFrame]
) -> T.List[T.Tuple[int, int]]:
    """Finds continuous segments of anomalies and returns a list of tuple with
    the start end end index of each anomaly.

    Args:
        y (T.union[np.ndarray, pd.DataFrame]): Array of labels (1d).

    Returns:
        T.List[T.Tuple[int, int]]: Returns a list of tuples. A tuple consists
        of 2 elements with the start and end index of the anomaly.


    Example:
        >>> find_anomalies(np.array([0, 1, 1, 0, 1, 0, 1, 1]))
        [(1, 2), (4, 4), (6, 7)]
    """

    y = np.array(y).squeeze()
    check_ndim(y, ndim=1)

    y = pd.Series(y)

    # true for index i when y[i - 1] = 0 and y[i] = 1
    start = (y > y.shift(1, fill_value=0))
    # true for index i when y[i] = 1 and y[i + 1] = 0
    end = (y > y.shift(-1, fill_value=0))

    # get indices where true
    start, end = np.flatnonzero(start), np.flatnonzero(end)

    return list(zip(start, end))


def recall_anomalies(
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    k: float = 0
) -> float:
    """Calculates the percentage of anomaly segments that are correctly
    detected. The parameter ``k`` [in %] controls how much of a segments needs
    to be detected for it being counted as detected.

    Args:
        y (np.ndarray): Ground Truth.
        y_hat (np.ndarray): Label predictions.
        k (float): Percentage ([0, 100]) of points that need to be detected in
          a segment for it to be counted. For K = 0, then at least one point
          has to be detected. For K = 100, then every point in the segment has
          to be correctly detected. Default: 0.

    Returns:
        float: Returns the fraction of detected anomaly segments.
    """
    y_hat, y = np.array(y_hat).squeeze(), np.array(y).squeeze()
    check_ndim(y_hat, y, ndim=1), check_shape(y_hat, y)

    if not (0 <= k <= 100):
        raise ValueError(f'k must be in [0, 100], got "{k}".')

    anomalies = find_anomalies(y)
    detected = _recall_anomalies(anomalies, y_hat, k=k)

    return detected / len(anomalies)


def _recall_anomalies(
    anomalies: T.List[T.Tuple[int, int]],
    y_hat: np.ndarray,
    *,
    k: float = 0
) -> int:
    """Determines the number of detected segments for a given ``k``.

    Args:
        anomalies (T.List[T.Tuple[int, int]]): Start and end index of
          anomalies.
        y_hat (np.ndarray): Predictions.
        k (float): Percentage ([0, 100]) of points that need to be detected in
          a segment for it to be counted. For K = 0, then at least one point
          has to be detected. For K = 100, then every point in the segment has
          to be correctly detected. Default: 0.

    Returns:
        int: Returns the number of detected segments.
    """
    # determine operator: for k = 0 use > else >=
    _op = gt if k == 0 else ge
    detected = np.sum([
        _op(np.sum(y_hat[s:(e + 1)]), ((k / 100) * (e + 1 - s)))
        for s, e in anomalies
    ])

    return detected


def rank_features(
    *,
    error: np.ndarray,
    y: np.ndarray
) -> T.Tuple[T.List[T.Tuple[int, int]], np.ndarray, np.ndarray]:
    """Finds the anomalies in y and calculates the feature-wise error for
    each anomaly. Each feature is ranked accordingly to their mean error
    during the anomaly.

    Args:
        error (np.ndarray): Error (2d or 3d).
        y (np.ndarray): Labels (1d).

    Raises:
        ValueError: Raised if length do not align for `error` and `y` or no
          anomalies were found.

    Returns:
        T.Tuple[T.List[T.Tuple[int, int]], np.ndarray]: Returns a tuple of 1.
        List of tuple where a tuple contains the start and end index of
        an anomaly. 2. A 2d array where each rows contains the ranked feature
        indexes. 3. A 2d array where each rows contains the mean error for the
        features in order of 2.

    Example:
        >>> errors = np.array([[0.1, 0.8, 0.3, 0.25], [0.2, 0.4, 0.2, 0.6]]).T
        >>> y = np.array([0, 1, 0, 1])
        >>> segments, rankings, mean_errors = rank_features(error=errors, y=y)
        >>> segments
            [(1, 1), (3, 3)]
        >>> rankings
            [[0, 1], [1, 0]]
        >>> mean_errors
            [[0.8, 0.4], [0.6, 0.25]]
    """

    # 1. finds all anomalies in y
    # 2. computes the average per feature per segment
    # 3. ranks the features according to their average error per anomaly

    error = detemporalize(error, verbose=False)
    y = detemporalize(y, verbose=False)

    if (e_len := error.shape[0]) != (y_len := y.shape[0]):
        warnings.warn(f'Length misaligned, got {e_len} and {y_len}.')

    (error, y), = truncate((error, y))

    if error.shape[-1] < 2:
        raise ValueError('Expected at least 2 features.')

    error = detemporalize(error)

    # get anomaly segments
    anomalies = find_anomalies(y)

    if len(anomalies) < 1:
        raise ValueError('No anomalies found.')

    # calculate mean per feature for each anomaly
    # rank the features according to their mean error for the anomaly
    errors = np.array([_sort_features(error, a) for a in anomalies])
    rankings = np.array(errors[:, :, 0], dtype='int32')
    mean_errors = errors[:, :, 1]

    return anomalies, rankings, mean_errors


def _sort_features(
        error: np.ndarray,
        idx: T.Tuple[int, int]
) -> T.List[T.Tuple[int, float]]:
    """Calculates the mean error per feature for an anomaly.

    Args:
        error (np.ndarray): Errors.
        idx (T.List[T.Tuple[int, int]]): Tuple of (start, end) indices of an
        anomaly.

    Returns:
        T.List[T.Tuple[int, float]]: Returns a list of sorted tuples containing
        the index and the mean error for the anomaly.
    """

    # calculate error by feature
    mean_err = np.mean(error[idx[0]:(idx[1] + 1)], axis=0)

    # rank error (tuple of (idx, mean_err))
    rank_err = sorted(enumerate(mean_err),
                      key=lambda item: item[1],
                      reverse=True)

    return rank_err


def apply_point_adjust(
    *,
    y_hat: np.ndarray,
    y: np.ndarray,
    k: float = 0
) -> np.ndarray:
    """Implements the point-adjust approach from
    https://arxiv.org/abs/1802.03903 and its variation from
    https://arxiv.org/abs/2109.05257 (parameter ``k``).
    For a ground truth anomaly segment in ``y``:

    - ``k=0``, if any point ``x`` in the segment was classified as
      anomalous (``y_hat=1`` for ``x``)
    - ``0 < k < 100``, if more than (>) ``%k`` of points in the segment
      are classified as anomalous (``y_hat=1`` for %k of points)
    - ``k=100`` if all points in the segment are classified as anomalous
      (``y_hat=1`` for all points)

    then the label for all observations in the segment are adjusted to
    ``y_hat=1``. If ``k=0`` it is equal to the original point-adjust, if
    ``k=100`` it is equal to the F1.

    Args:
        y_hat (np.ndarray): Label predictions (1d).
        y (np.ndarray): Ground Truth (1d).
        k (int): Percentage [0, 100] of points detected as an anomaly in a
          segment before an adjustment is made (Default: 0).

    Returns:
        np.ndarray: Returns the point-adjusted ``y_hat``.

    Example:
        >>> import mlnext
        >>> import numpy as np
        >>> mlnext.apply_point_adjust(
        ...   y_hat = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1]),
        ...   y =     np.array([0, 0, 1, 1, 1, 0, 1, 1, 0]))
        [1, 0, 1, 1, 1, 0, 1, 1, 1]

        >>> # for k = 40; only adjusts the second segment
        >>> mlnext.apply_point_adjust(
        ...   y_hat = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1]),
        ...   y =     np.array([0, 0, 1, 1, 1, 0, 1, 1, 0])
        ...   k = 40)
        [1, 0, 0, 1, 0, 0, 1, 1, 1]

    """
    y, y_hat = np.array(y).squeeze(), np.array(y_hat).squeeze()
    check_ndim(y, y_hat, ndim=1)

    if y_hat.shape != y.shape:
        warnings.warn(f'Shapes unaligned {y_hat.shape} and {y.shape}.')

    (y_hat, y), = truncate((y_hat, y))
    y_hat = np.copy(y_hat)

    if k < 0 or k > 100:
        raise ValueError(f'Parameter k must be in [0, 100], but got: {k}.')

    for (start, end) in find_anomalies(y):
        s = np.s_[start: (end + 1)]

        # check if more than %k points of that segment are anomalous
        # otherwise the label is left as is
        if np.sum(y_hat[s]) > (k * (end + 1 - start)) / 100:
            y_hat[s] = 1

    return y_hat


def apply_point_adjust_score(
    *,
    y_score: np.ndarray,
    y: np.ndarray,
    k: float = 0
) -> np.ndarray:
    """Implements the point-adjust approach from
    https://arxiv.org/pdf/1802.03903.pdf  and its variation from
    https://arxiv.org/abs/2109.05257 (parameter ``k``) for prediction scores.
    For a ground truth anomaly segment in ``y``:

    - ``k=0``, the score of all points are adjusted to the maximum score in
      the segment
    - ``0 < k < 100``, the score for the adjustment is chosen, such that at
      least %k of points in the anomaly segments have a higher score and
      only  the points below the chosen score are adjusted to the score
    - ``k=100``, no adjustment is made

    If ``k=0`` it is equal to the original point-adjust, if ``k=100`` it is
    equal to the F1. This method allows the usage of the point-adjust method
    in conjunction with precision-recall and other similar curves.

    Args:
        y_score (np.ndarray): Prediction score in range [0, 1] (1d array).
        y (np.ndarray): Ground truth (1d array).

    Returns:
        np.ndarray: Returns the adjusted array.

    Example:
        >>> import numpy as np
        >>> import mlnext
        >>> mlnext.apply_point_adjust_score(
        ... y_score = np.array([0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25]),
        ... y=        np.array([  0,   0,   1,   1,   1,   0,   1,   1,    0]),
        ... k=0)
        [0.1, 0.4, 0.7, 0.7, 0.7, 0.2, 0.6, 0.6, 0.25]

        >>> # for k=40; both segments are adjusted
        >>> mlnext.apply_point_adjust_score(
        ... y_score = np.array([0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25]),
        ... y=        np.array([  0,   0,   1,   1,   1,   0,   1,   1,    0]),
        ... k=40)
        [0.1, 0.4, 0.6, 0.7, 0.6, 0.2, 0.6, 0.6, 0.25]
    """
    y, y_score = np.array(y).squeeze(), np.array(y_score).squeeze()
    check_ndim(y, y_score, ndim=1)

    if y_score.shape != y.shape:
        warnings.warn(f'Shapes unaligned {y_score.shape} and {y.shape}.')

    (y_score, y), = truncate((y_score, y))
    y_score = np.copy(y_score)

    if k < 0 or k > 100:
        raise ValueError(f'Parameter k must be in [0, 100], but got: {k}.')

    for (start, end) in find_anomalies(y):
        s = np.s_[start: (end + 1)]

        # find the index of the element that fulfills the condition:
        # at least %k points are above a threshold
        length = (end + 1) - start

        index = min(int(np.floor((length * k) / 100)) + 1, length)
        score = np.sort(y_score[s])[-index]
        # adjust only points that are below the score
        mask = y_score[s] < score
        y_score[s][mask] = score

    return y_score
