import typing as T

import numpy as np
import pytest

from mlnext.anomaly import apply_point_adjust
from mlnext.anomaly import apply_point_adjust_score
from mlnext.anomaly import find_anomalies
from mlnext.anomaly import hit_rate_at_p
from mlnext.anomaly import rank_features
from mlnext.anomaly import recall_anomalies
from mlnext.score import apply_threshold


@pytest.mark.parametrize(
    'y,exp',
    [
        ([0, 1, 1, 0, 1, 0, 1, 1], [(1, 2), (4, 4), (6, 7)]),
        ([1, 1, 0, 1, 1], [(0, 1), (3, 4)]),
        ([0, 1, 0, 1, 0], [(1, 1), (3, 3)]),
        ([0, 0, 0, 0, 0], []),
        ([1, 1, 1, 1], [(0, 3)]),
        ([1], [(0, 0)]),
        (np.array([[0, 1, 1, 0, 1, 0, 1, 1]]).T, [(1, 2), (4, 4), (6, 7)]),
    ],
)
def test_find_anomalies(y: np.ndarray, exp: T.List[T.Tuple[int, int]]):
    result = find_anomalies(y=np.array(y))

    assert result == exp


@pytest.mark.parametrize(
    'y,err_msg',
    [
        (
            [[0, 1], [1, 0]],
            'Expected array of dimension 1, but got 2 for array at position 0.',
        ),
    ],
)
def test_find_anomalies_fails(y: np.ndarray, err_msg: str):
    with pytest.raises(ValueError) as exc_info:
        find_anomalies(y=np.array(y))

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'error,y,exp',
    [
        (
            [[0.1, 0.6, 0.6, 0.4, 0.5], [0.05, 0.3, 0.4, 0.3, 0.7]],
            [0, 1, 1, 0, 1],
            ([(1, 2), (4, 4)], [[0, 1], [1, 0]], [[0.6, 0.35], [0.7, 0.5]]),
        ),
        (
            [[0.1, 0.8, 0.3, 0.25], [0.2, 0.4, 0.2, 0.6]],
            [0, 1, 0, 1],
            ([(1, 1), (3, 3)], [[0, 1], [1, 0]], [[0.8, 0.4], [0.6, 0.25]]),
        ),
    ],
)
def test_rank_features(error: np.ndarray, y: np.ndarray, exp: T.Tuple):
    error = np.array(error).T
    y = np.array(y)

    result = rank_features(error=error, y=y)

    np.testing.assert_equal(result, exp)


@pytest.mark.parametrize(
    'error,y,reduce,err_msg',
    [
        (
            [[0.1, 0.6, 0.4, 0.5]],
            [0, 1, 1, 1],
            'mean',
            'Expected at least 2 features.',
        ),
        (
            [[0.1, 0.2], [0.1, 0.3]],
            [0, 0, 0, 0],
            'mean',
            'No anomalies found.',
        ),
        (
            [[0.1, 0.2], [0.1, 0.3]],
            [0, 1],
            '1',
            'Received invalid value "1" for reduce. Available functions are: '
            "['mean', 'max', 'sum'].",
        ),
    ],
)
def test_rank_features_fails(
    error: np.ndarray,
    y: np.ndarray,
    reduce: str,
    err_msg: str,
):
    error = np.array(error).T
    y = np.array(y)

    with pytest.raises(ValueError) as exc_info:
        rank_features(error=error, y=y, reduce=reduce)

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'reduce,mask,exp',
    [
        (
            'mean',
            None,
            ([(1, 3), (5, 5)], [[0, 1], [1, 0]], [[0.6, 0.3], [0.8, 0.1]]),
        ),
        (
            'mean',
            [0, 0, 0, 1, 0, 0],
            ([(1, 3), (5, 5)], [[1, 0], [0, 1]], [[0.6, 0.5], [0.0, 0.0]]),
        ),
        (
            'max',
            None,
            ([(1, 3), (5, 5)], [[0, 1], [1, 0]], [[0.7, 0.6], [0.8, 0.1]]),
        ),
        (
            'max',
            [0, 0, 0, 1, 0, 0],
            ([(1, 3), (5, 5)], [[1, 0], [0, 1]], [[0.6, 0.5], [0.0, 0.0]]),
        ),
        (
            'sum',
            None,
            ([(1, 3), (5, 5)], [[0, 1], [1, 0]], [[1.8, 0.9], [0.8, 0.1]]),
        ),
        (
            'sum',
            [0, 0, 0, 1, 0, 0],
            ([(1, 3), (5, 5)], [[1, 0], [0, 1]], [[0.6, 0.5], [0.0, 0.0]]),
        ),
    ],
)
@pytest.mark.parametrize(
    'error,y',
    [
        (
            [[0.1, 0.6, 0.7, 0.5, 0.5, 0.1], [0.05, 0.1, 0.2, 0.6, 0.7, 0.8]],
            [0, 1, 1, 1, 0, 1],
        ),
    ],
)
def test_rank_features_reduce(
    error: np.ndarray,
    y: np.ndarray,
    reduce: str,
    mask: T.Optional[np.ndarray],
    exp: np.ndarray,
):
    error = np.array(error).T
    y = np.array(y)
    mask = np.array(mask).astype('bool') if mask is not None else None

    result = rank_features(
        error=error,
        y=y,
        reduce=reduce,
        mask=mask,
    )
    np.testing.assert_almost_equal(result, exp)


@pytest.mark.parametrize(
    'y_hat,y,k,exp',
    [
        (
            [1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
            [1, 0, 1, 1, 1, 0, 0, 0, 1],
        ),
        (
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
            0,
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        ),
        (
            [1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
            [1, 0, 1, 1, 1, 0, 1, 1, 1],
        ),
        (
            [1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
            [1, 0, 1, 1, 1, 0, 1, 1, 1],
        ),
        (
            [1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            40,
            [1, 0, 0, 1, 0, 0, 1, 1, 1],
        ),
    ],
)
def test_apply_point_adjust(
    y_hat: np.ndarray, y: np.ndarray, k: int, exp: np.ndarray
):
    result = apply_point_adjust(y_hat=np.array(y_hat), y=np.array(y), k=k)

    np.testing.assert_array_equal(result, exp)


@pytest.mark.parametrize('k', list(range(1, 100)))
def test_apply_point_adjust_k_1_99(k: int):
    y_hat = np.r_[[1] * (k + 1), [0] * (100 - k - 1)]
    y = np.ones(100)

    result = apply_point_adjust(y_hat=y_hat, y=y, k=k)
    np.testing.assert_array_equal(result, y)


@pytest.mark.parametrize('k', list(range(1, 100)))
def test_apply_point_adjust_k_1_99_not_adjusted(k: int):
    y_hat = np.r_[[1] * k, [0] * (100 - k)]
    y = np.ones(100)

    result = apply_point_adjust(y_hat=y_hat, y=y, k=k)
    np.testing.assert_array_equal(result, y_hat)


@pytest.mark.parametrize(
    'y_score,y,k,exp',
    [
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            0,
            [0.1, 0.4, 0.6, 0.7, 0.7, 0.7, 0.4, 0.6, 0.25],
        ),
        (
            [0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.4, 0.4, 0.2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            0,
            [0.6] * 9,
        ),
        (
            [0.2, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            0,
            [0.2, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2],
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
            [0.1, 0.4, 0.7, 0.7, 0.7, 0.2, 0.6, 0.6, 0.25],
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            40,
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.4, 0.4, 0.6, 0.25],
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            40,
            [0.1, 0.4, 0.6, 0.7, 0.6, 0.2, 0.6, 0.6, 0.25],
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            100,
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
        ),
    ],
)
def test_apply_point_adjust_score(
    y_score: np.ndarray, y: np.ndarray, k: float, exp: np.ndarray
):
    result = apply_point_adjust_score(
        y_score=np.array(y_score), y=np.array(y), k=k
    )

    np.testing.assert_array_equal(result, exp)


@pytest.mark.parametrize('k', list(range(1, 100)))
def test_apply_point_adjust_score_k_1_100(
    k: int, y_score=np.arange(0, 1, 0.01), y=np.ones(100)
):
    exp = np.copy(y_score)
    exp[: -(k + 1)] = exp[-(k + 1)]

    result = apply_point_adjust_score(y_score=y_score, y=y, k=k)

    np.testing.assert_array_equal(result, exp)


@pytest.mark.parametrize(
    'y_score,y,k',
    [
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            0,
        ),
        (
            [0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.4, 0.4, 0.2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            0,
        ),
        (
            [0.2, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            0,
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            0,
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            0,
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            40,
        ),
        (
            [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            100,
        ),
    ],
)
def test_apply_point_adjust_threshold(
    y_score: np.ndarray, y: np.ndarray, k: float
):
    y_score = np.array(y_score)
    y = np.array(y)

    y_score = apply_point_adjust_score(y_score=y_score, y=y, k=k)
    y_score = apply_threshold(y_score, threshold=0.5)

    y_pred = apply_threshold(y_score, threshold=0.5)
    y_pred = apply_point_adjust(y_hat=y_pred, y=y, k=k)

    np.testing.assert_array_equal(y_score, y_pred)


@pytest.mark.parametrize('k', list(range(0, 101)))
def test_apply_point_adjust_threshold_k_0_100(
    k: float,
    y_score: np.ndarray = np.array(
        [0.1, 0.4, 0.6, 0.7, 0.4, 0.2, 0.4, 0.6, 0.25]
    ),
    y: np.ndarray = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0]),
):
    y_score = apply_point_adjust_score(y_score=y_score, y=y, k=k)
    y_score = apply_threshold(y_score, threshold=0.5)

    y_pred = apply_threshold(y_score, threshold=0.5)
    y_pred = apply_point_adjust(y_hat=y_pred, y=y, k=k)

    np.testing.assert_array_equal(y_score, y_pred)


@pytest.mark.parametrize(
    'y_hat,y,k,exp',
    [
        ([0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0], 0, 1.0),
        ([0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0], 60, 0.0),
        ([0, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0], 50, 1.0),
        ([0, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0], 60, 0.5),
        ([0, 1, 1, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 1, 1, 0], 100, 1.0),
    ],
)
def test_recall_anomalies(
    y: np.ndarray, y_hat: np.ndarray, k: float, exp: float
):
    result = recall_anomalies(y, y_hat, k=k)

    assert result == exp


@pytest.mark.parametrize(
    'y_hat,y,k,msg',
    [
        ([0, 1], [0, 1], -1, 'k must be in [0, 100], got "-1".'),
        ([0, 1], [0, 1], 101, 'k must be in [0, 100], got "101".'),
    ],
)
def test_recall_anomalies_fails(
    y: np.ndarray, y_hat: np.ndarray, k: float, msg: str
):
    with pytest.raises(ValueError) as ex:
        recall_anomalies(y, y_hat, k=k)

    assert ex.value.args[0] == msg


@pytest.mark.parametrize(('p'), (100, 150))
@pytest.mark.parametrize(
    ('dims', 'ranking', 'exp'),
    (
        ([2], [1, 3, 6, 2, 5, 4], [0.0, 0.0]),
        ([2], [2, 3, 6, 1, 5, 4], [1.0, 1.0]),
        ([2, 6], [5, 3, 1, 4, 2, 6], [0.0, 0.0]),
        ([3, 6], [5, 2, 3, 4, 1, 6], [0.0, 0.5]),
        ([6, 2], [6, 5, 1, 4, 3, 2], [0.5, 0.5]),
        ([2, 6], [2, 6, 4, 1, 2, 3], [1.0, 1.0]),
        ([1, 6, 4], [3, 8, 7, 2, 1, 6, 5, 4], [0.0, 0.0]),
        ([3, 4, 8], [4, 6, 1, 2, 7, 8, 5, 3], [0.33, 0.33]),
        ([3, 4, 8], [4, 6, 1, 8, 7, 2, 5, 3], [0.33, 0.66]),
        ([1, 6, 4], [4, 6, 1, 2, 7, 8, 5, 3], [1.0, 1.0]),
        ([], [2, 3, 1, 4], [0.0, 0.0]),
        ([2, 3, 1, 4], [2, 3, 1, 4], [1.0, 1.0]),
        ([2, 3, 1, 4], [2, 3, 1], [0.75, 0.75]),
    ),
)
def test_hit_rate_at_p(
    dims: np.ndarray,
    ranking: np.ndarray,
    p: T.Literal[100, 150],
    exp: T.List[float],
):
    expected: float = exp[0] if p == 100 else exp[1]

    result = hit_rate_at_p(dims, ranking, p=p)

    np.testing.assert_almost_equal(result, expected, decimal=2)
