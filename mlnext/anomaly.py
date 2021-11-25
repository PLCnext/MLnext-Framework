"""Module for analyzing anomalies.
"""
import typing as T
import warnings

import numpy as np
import pandas as pd

from .data import detemporalize


def find_anomalies(
    y: T.Union[np.array, pd.DataFrame]
) -> T.List[T.Tuple[int, int]]:
    """Finds continuous segments of anomalies and returns a list of tuple with
    the start end end index of each anomaly.

    Args:
        y (T.union[np.array, pd.DataFrame]): Array of labels (1d).

    Returns:
        T.List[T.Tuple[int, int]]: Returns a list of tuples. A tuple consists
        of 2 elements with the start and end index of the anomaly.


    Example:
        >>> find_anomalies(np.array([0, 1, 1, 0, 1, 0, 1, 1]))
        [(1, 2), (4, 4), (6, 7)]
    """

    y = y.squeeze()
    if len(y.shape) > 1:
        raise ValueError(f'Expected y to be an 1d array, but got {y.shape}.')

    y = pd.DataFrame(y)

    # true for index i when y[i - 1] = 0 and y[i] = 1
    start = (y > y.shift(1, fill_value=0))
    # true for index i when y[i] = 1 and y[i + 1] = 0
    end = (y > y.shift(-1, fill_value=0))

    # get indices where true
    start, end = np.flatnonzero(start), np.flatnonzero(end)

    return list(zip(start, end))


def rank_features(
    *,
    error: np.array,
    y: np.array
) -> T.Tuple[T.List[T.Tuple[int, int]], np.array, np.array]:
    """Finds the anomalies in y and calculates the feature-wise error for
    each anomaly. Each feature is ranked accordingly to their mean error
    during the anomaly.

    Args:
        error (np.array): Error (2d or 3d).
        y (np.array): Labels (1d).

    Raises:
        ValueError: Raised if length do not align for `error` and `y` or no
          anomalies were found.

    Returns:
        T.Tuple[T.List[T.Tuple[int, int]], np.array]: Returns a tuple of 1.
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

    error, y = _truncate_arrays(error, y)

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
        error: np.array,
        idx: T.Tuple[int, int]
) -> T.List[T.Tuple[int, float]]:
    """Calculates the mean error per feature for an anomaly.

    Args:
        error (np.array): Errors.
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


def apply_point_adjust(*, y_hat: np.array, y: np.array) -> np.array:
    """Implements the point-adjust approach from
    https://arxiv.org/pdf/1802.03903.pdf. For any observation
    in the ground truth anomaly segment in `y`, is detected as anomaly in
    `y_hat`, then the segment is detected correctly and the label for each
    observation in the anomaly segment is set to 1.

    Args:
        y_hat (np.array): Label predictions (1d).
        y (np.array): Ground Truth (1d).

    Returns:
        np.array: Returns the point-adjusted `y_hat`.
    """
    if y_hat.shape != y.shape:
        warnings.warn(f'Shapes unaligned {y_hat.shape} and {y.shape}.')

    y_hat, y = _truncate_arrays(y_hat, y)

    y_hat = np.copy(y_hat)
    for (start, end) in find_anomalies(y):
        # check if y_hat has any observation that lies in the
        # anomaly segment from start to end
        if np.any(y_hat[start:(end + 1)]):
            y_hat[start:(end + 1)] = 1

    return y_hat


def apply_point_adjust_score(
    *,
    y_score: np.array,
    y: np.array
) -> np.array:
    """Implements the point-adjust approach from
    https://arxiv.org/pdf/1802.03903.pdf for prediction scores.
    Thus, the point-adjust method can be used for precision-recall and other
    similar curves. For any in the ground truth anomaly segment in `y`,
    the score `y_score` is set to the maximum score in the anomaly segment.

    Args:
        y_score (np.array): Prediction (usually in range [0, 1]).
        y (np.array): Ground truth.

    Returns:
        np.array: Returns the adjusted array.
    """

    if y_score.shape != y.shape:
        warnings.warn(f'Shapes unaligned {y_score.shape} and {y.shape}.')

    y_score, y = _truncate_arrays(y_score, y)

    y_score = np.copy(y_score)
    for (start, end) in find_anomalies(y):
        # set the error for the anomaly to the maximum score
        y_score[start:(end + 1)] = np.max(y_score[start:(end + 1)])

    return y_score


def _truncate_arrays(*arrays: np.array) -> T.List[np.array]:
    """Truncates a list of arrays to the same length.

    Args:
        arrays (List[np.array]): List of arrays.

    Returns:
        T.List[np.array]: Returns the list of arrays truncated to the length of
        the shortest array in the list.
    """
    length = min([arr.shape[0] for arr in arrays])
    return [arr[:length] for arr in arrays]
