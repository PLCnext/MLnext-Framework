import typing as T

import numpy as np
import pytest

from mlnext.anomaly import _truncate_arrays
from mlnext.anomaly import apply_point_adjust
from mlnext.anomaly import apply_point_adjust_score
from mlnext.anomaly import find_anomalies
from mlnext.anomaly import rank_features
from mlnext.score import apply_threshold


@pytest.mark.parametrize(
    'y,exp',
    [
        ([0, 1, 1, 0, 1, 0, 1, 1], [(1, 2), (4, 4), (6, 7)]),
        ([1, 1, 0, 1, 1], [(0, 1), (3, 4)]),
        ([0, 1, 0, 1, 0], [(1, 1), (3, 3)]),
        ([0, 0, 0, 0, 0], []),
        ([1, 1, 1, 1], [(0, 3)]),

        (np.array([[0, 1, 1, 0, 1, 0, 1, 1]]).T, [(1, 2), (4, 4), (6, 7)])
    ]
)
def test_find_anomalies(y: np.array, exp: T.List[T.Tuple[int, int]]):

    result = find_anomalies(y=np.array(y))

    assert result == exp


@pytest.mark.parametrize(
    'y,err_msg',
    [
        ([[0, 1], [1, 0]],
         'Expected y to be an 1d array, but got (2, 2).'),
    ]
)
def test_find_anomalies_fails(y: np.array, err_msg: str):

    with pytest.raises(ValueError) as exc_info:
        find_anomalies(y=np.array(y))

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'error,y,exp',
    [
        ([[0.1, 0.6, 0.6, 0.4, 0.5], [0.05, 0.3, 0.4, 0.3, 0.7]],
         [0, 1, 1, 0, 1],
         ([(1, 2), (4, 4)],
          [[0, 1], [1, 0]],
          [[0.6, 0.35], [0.7, 0.5]])),

        ([[0.1, 0.8, 0.3, 0.25], [0.2, 0.4, 0.2, 0.6]],
         [0, 1, 0, 1],
         ([(1, 1), (3, 3)], [[0, 1], [1, 0]], [[0.8, 0.4], [0.6, 0.25]]))
    ]
)
def test_rank_features(error: np.array, y: np.array, exp: T.Tuple):

    error = np.array(error).T
    y = np.array(y)

    result = rank_features(error=error, y=y)

    np.testing.assert_equal(result, exp)


@pytest.mark.parametrize(
    'error,y,err_msg',
    [
        ([[0.1, 0.6, 0.4, 0.5]],
         [0, 1, 1, 1], 'Expected at least 2 features.'),

        ([[0.1, 0.2], [0.1, 0.3]],
         [0, 0, 0, 0], 'No anomalies found.'),
    ]
)
def test_rank_features_fails(error: np.array, y: np.array, err_msg: str):

    error = np.array(error).T
    y = np.array(y)

    with pytest.raises(ValueError) as exc_info:
        rank_features(error=error, y=y)

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'y_hat,y,exp',
    [
        ([1, 0, 0, 1, 0, 0, 0, 0, 1],
         [0, 0, 1, 1, 1, 0, 1, 1, 0],
         [1, 0, 1, 1, 1, 0, 0, 0, 1]),

        ([1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 0, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]),

        ([0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]),

        ([1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
         [1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    ]

)
def test_apply_point_adjust(
    y_hat: np.array,
    y: np.array,
    exp: np.array
) -> np.array:

    result = apply_point_adjust(y_hat=np.array(y_hat), y=np.array(y))

    np.testing.assert_array_equal(result, exp)


@pytest.mark.parametrize(
    'arrays,exp',
    [
        ([np.zeros((15, 1)), np.ones((10, 2))],
         [np.zeros((10, 1)), np.ones((10, 2))])

    ]
)
def test_truncate_arrays(arrays: T.List[np.array], exp: T.List[np.array]):

    result = _truncate_arrays(*arrays)

    np.testing.assert_equal(result, exp)


@pytest.mark.parametrize(
    'y_score,y,exp',
    [
        ([0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
         [1, 0, 0, 1, 1, 1, 0, 0, 1],
         [0.1, 0.4, 0.6, 0.7, 0.7, 0.7, 0.4, 0.6, 0.25]),

        ([0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.4, 0.4, 0.2],
         [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0.6] * 9),

        ([0.2, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0.2, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2])
    ]

)
def test_apply_point_adjust_score(
    y_score: np.array,
    y: np.array,
    exp: np.array
) -> np.array:
    result = apply_point_adjust_score(y_score=np.array(y_score), y=np.array(y))

    np.testing.assert_array_equal(result, exp)


@pytest.mark.parametrize(
    'y_score,y',
    [
        ([0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
         [1, 0, 0, 1, 1, 1, 0, 0, 1]),

        ([0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.4, 0.4, 0.2],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]),

        ([0.2, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2],
         [0, 0, 0, 0, 0, 0, 0, 0, 0])
    ]

)
def test_apply_point_adjust_threshold(
    y_score: np.array,
    y: np.array
) -> np.array:
    y = np.array(y)

    y_score = apply_point_adjust_score(y_score=np.array(y_score), y=y)
    y_score = apply_threshold(y_score, threshold=0.5)

    y_pred = apply_threshold(y_score, threshold=0.5)
    y_pred = apply_point_adjust(y_hat=y_pred, y=y)

    np.testing.assert_array_equal(y_score, y_pred)
