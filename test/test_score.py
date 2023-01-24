import typing as T
from dataclasses import fields
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from mlnext import score
from mlnext.score import _recall_anomalies_curve
from mlnext.score import auc_point_adjust_metrics
from mlnext.score import kl_divergence
from mlnext.score import point_adjust_metrics
from mlnext.score import pr_curve
from mlnext.score import PRCurve


class TestL2Norm(TestCase):

    def test_l2_norm(self):

        x = np.arange(10).reshape(-1, 2)
        x_hat = np.arange(10).reshape(-1, 2) + 1

        expected = np.ones((5, 1)) * np.sqrt(2.0)
        result = score.l2_norm(x, x_hat)

        np.testing.assert_array_almost_equal(result, expected)

    def test_l2_norm_feature_wise(self):

        x = np.arange(10).reshape(-1, 2)
        x_hat = np.arange(10).reshape(-1, 2) + 1

        expected = np.ones((5, 2))
        result = score.l2_norm(x, x_hat, reduce=False)

        np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    'p,exp',
    [
        (100, 1.),
        (80, 0.8),
        (50, 0.5),
        (10, 0.1),
        (0, 0.0)
    ]
)
def test_get_threshold(p: float, exp: float):

    result = score.get_threshold(np.array([0, 1]), p=p)

    assert result == exp


@pytest.mark.parametrize(
    'x,threshold,pos_label,exp',
    [
        ([0, 0.4, 0.6, 1.], 0.5, 1,  [0, 0, 1, 1]),
        ([0, 0.4, 0.6, 1.], 0.5, 0,  [1, 1, 0, 0]),
    ]
)
def test_apply_threshold(
    x: T.List[float],
    threshold: float,
    pos_label: int,
    exp: T.List[float]
):

    result = score.apply_threshold(
        np.array(x),
        threshold=threshold,
        pos_label=pos_label
    )

    np.testing.assert_array_equal(result, exp)


class TestEval(TestCase):

    def test_eval_softmax(self):

        data = np.array([0.2, 0.3, 0.5])
        expected = np.array([[2]], dtype=np.int64)

        result = score.eval_softmax(data)

        np.testing.assert_array_equal(result, expected)

    def test_eval_softmax_2d(self):

        data = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
        expected = np.array([[2], [1]], dtype=np.int64)

        result = score.eval_softmax(data)

        np.testing.assert_array_equal(result, expected)

    def test_eval_softmax_3d(self):

        data = np.array([[[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]]])
        expected = np.array([[2], [1]], dtype=np.int64)

        result = score.eval_softmax(data)

        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'y,invert,threshold,exp',
    [
        ([[0.2], [0.6]], False, 0.5, [[0], [1]]),
        ([[[0.2], [0.6]]], False, 0.5, [[0], [1]]),
        ([[0.2], [0.6]], True, 0.5, [[1], [0]]),
        ([[[0.2], [0.6]]], True, 0.5, [[1], [0]]),

        (np.linspace(0, 1, 11), False, 0.5,
         np.r_[np.zeros((5, 1)), np.ones((6, 1))]),

        (np.linspace(0, 1, 11), True, 0.5, np.r_[
         np.ones((5, 1)), np.zeros((6, 1))])
    ]
)
def test_eval_sigmoid(
    y: T.List[float],
    invert: bool,
    threshold: float,
    exp: T.List[float]
):

    print(y)

    result = score.eval_sigmoid(
        np.array(y),
        invert=invert,
        threshold=threshold
    )

    np.testing.assert_array_equal(result, exp)


class TestMovingAverage(TestCase):

    def test_moving_average(self):

        data = np.arange(5)
        expected = [0, 0.5, 1.5, 2.5, 3.5, 2]
        result = score.moving_average(data, 2, mode='full')

        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'y,y_hat',
    [
        (np.ones(10), np.ones(10)),
        (np.ones((10, 1)), np.ones((10, 1))),
    ]
)
def test_eval_metrics(
    y: np.ndarray,
    y_hat: np.ndarray
):
    exp = {
        'accuracy': 1.0,
        'precision': 1.0,
        'recall': 1.0,
        'f1': 1.0,
        'anomalies': 1.0
    }
    result = score.eval_metrics(y, y_hat)

    assert result == exp


@pytest.mark.parametrize(
    'y,y_hat',
    [
        (np.ones(10), np.ones(12)),
        (np.ones(12), np.ones(10)),
        (np.ones((10, 1)), np.ones((12, 1))),
        (np.ones((12, 1)), np.ones((10, 1)))
    ]
)
def test_eval_metrics_warns(
    y: np.ndarray,
    y_hat: np.ndarray
):
    with pytest.warns(UserWarning) as record:
        score.eval_metrics(y, y_hat)

    msg = f'Shapes unaligned y {y.shape} and y_hat {y_hat.shape}.'
    assert record[0].message.args[0] == msg


@pytest.mark.parametrize(
    'y,y_hat',
    [
        ([np.ones(10), np.zeros(10)], [np.ones(10), np.zeros(10)]),
        ([np.ones((10, 1)), np.zeros(10)], [np.ones((10, 1)), np.zeros(10)]),
        (
            [np.ones(10), np.zeros(10), np.ones(10)],
            [np.ones(10), np.zeros(10), np.ones(10)]
        ),
    ]
)
def test_eval_metrics_all(
    y: np.ndarray,
    y_hat: np.ndarray
):
    exp = {
        'accuracy': 1.0,
        'precision': 1.0,
        'recall': 1.0,
        'f1': 1.0,
        'anomalies': 1.0
    }
    result = score.eval_metrics_all(y, y_hat)

    assert result == exp


@pytest.mark.parametrize(
    'y,y_hat,msg',
    [
        (
            [np.ones(10), np.ones(10)], [np.ones(10)],
            'y and y_hat must have the same number elements, '
            'but found y=2 and y_hat=1.'
        ),
        (
            [np.ones((10, 2))], [np.ones(10)],
            'Expected axis 1 of array to be of size 1, '
            'but got 2 for array at position 0.'
        ),
        (
            [np.ones((10, 1))], [np.ones((10, 2))],
            'Expected axis 1 of array to be of size 1, '
            'but got 2 for array at position 0.'
        ),
        (
            [np.ones((10, 1, 1))], [np.ones((10, 1))],
            'Expected array of dimension 2, but got 3 for array at position 0.'
        ),
        (
            [np.ones((10, 1))], [np.ones((10, 1, 1))],
            'Expected array of dimension 2, but got 3 for array at position 0.'
        ),
    ]
)
def test_eval_metrics_all_raises(
    y: np.ndarray,
    y_hat: np.ndarray,
    msg: str
):

    with pytest.raises(ValueError) as exc_info:
        score.eval_metrics_all(y, y_hat)

    assert exc_info.value.args[0] == msg


class TestNLL(TestCase):

    def test_norm_log_likelihood(self):

        mu = np.array([[1.0, 2.0, 3.0]]).reshape(-1, 1)
        var = np.array([[0.1, 0.5, 0.8]]).reshape(-1, 1)
        x = np.array([[1.0, 1.4, 3.2]]).reshape(-1, 1)

        expected = -scipy.stats.norm.logpdf(x, mu, np.sqrt(var))

        result = score.norm_log_likelihood(x, mu, np.log(var))

        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_bern_log_likelihood(self):

        mu = np.array([[0.2, 0.34, 0.89, 0.0]],
                      dtype=np.float32).reshape(-1, 1)
        x = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32).reshape(-1, 1)
        expected = -scipy.stats.bernoulli.logpmf(x, mu)

        result = score.bern_log_likelihood(x, mu)

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    'mean,log_var,prior_mean,prior_std,exp',
    [
        (np.array([0.0]), np.array([0.0]), 0.0, 1.0, 0.0),
        (np.array([1.0]), np.array([np.log(0.1**2)]), 1.0, 0.1, 0.0),
    ]
)
def test_kl_divergence(
    mean: np.ndarray,
    log_var: np.ndarray,
    prior_mean: float,
    prior_std: float,
    exp: np.ndarray
):

    np.testing.assert_array_almost_equal(kl_divergence(
        mean=mean,
        log_var=log_var,
        prior_mean=prior_mean,
        prior_std=prior_std
    ), exp)


@pytest.mark.parametrize(
    'y_true,y_score,exp',
    [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8],
         PRCurve(
            tps=np.array([2., 1., 1.]),
            fns=np.array([0., 1., 1.]),
            tns=np.array([1., 1., 2.]),
            fps=np.array([1., 1., 0.]),
            das=np.array([1., 1., 1.]),
            tas=np.array([1., 1., 1.]),

            precision=np.array([0.6667, 0.5000, 1.0000]),
            recall=np.array([1.0000, 0.5000, 0.5000]),
            thresholds=np.array([0.35, 0.4, 0.8])

        )),
        ([0, 0, 1, 1, 1], [0.1, 0.4, 0.5, 0.35, 0.8],
         PRCurve(
            tps=np.array([3., 2., 2., 1.]),
            fns=np.array([0., 1., 1., 2.]),
            tns=np.array([1., 1., 2., 2.]),
            fps=np.array([1., 1., 0., 0.]),
            das=np.array([1., 1., 1., 1.]),
            tas=np.array([1., 1., 1., 1.]),

            precision=np.array([0.75, 0.6667, 1.0000, 1.0000]),
            recall=np.array([1.0000, 0.6667, 0.6667, 0.3333]),
            thresholds=np.array([0.35, 0.4, 0.5, 0.8])

        ))
    ]
)
def test_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exp: PRCurve
):

    result = pr_curve(np.array(y_true), np.array(y_score))

    for field in fields(exp):
        r, e = getattr(result, field.name), getattr(exp, field.name)
        np.testing.assert_almost_equal(r, e, decimal=4)

    # confusion matrix
    for i, cm in enumerate(result):
        np.testing.assert_almost_equal(cm.TP, exp.tps[i])
        np.testing.assert_almost_equal(cm.FN, exp.fns[i])
        np.testing.assert_almost_equal(cm.TN, exp.tns[i])
        np.testing.assert_almost_equal(cm.FP, exp.fps[i])

        np.testing.assert_almost_equal(
            cm.precision, exp.precision[i], decimal=4)
        np.testing.assert_almost_equal(cm.recall, exp.recall[i], decimal=4)
        np.testing.assert_almost_equal(cm.f1, exp.f1[i], decimal=4)
        np.testing.assert_almost_equal(cm.accuracy, exp.accuracy[i], decimal=4)


@pytest.mark.parametrize(
    'y_hat,y',
    [
        ([0, 1, 1, 0], [0, 1, 1, 0])
    ]
)
def test_point_adjust_metrics(
    y_hat: np.ndarray,
    y: np.ndarray,
):
    result = point_adjust_metrics(y_hat=np.array(y_hat), y=np.array(y))

    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    'y_hat,y,exp',
    [
        (
            [0, 1, 1, 0], [0, 1, 1, 0],
            {
                'auc_accuracy': 1.0,
                'auc_precision': 1.0,
                'auc_recall': 1.0,
                'auc_f1': 1.0,
                'auc_anomalies': 1.0
            }
        )
    ]
)
def test_auc_point_adjust_metrics(
    y_hat: np.ndarray,
    y: np.ndarray,
    exp: T.Dict[str, float]
):
    result = auc_point_adjust_metrics(y_hat=np.array(y_hat), y=np.array(y))

    assert isinstance(result, dict)
    np.testing.assert_equal(result, exp)


@pytest.mark.parametrize(
    'y_score,thresholds,pos_label,k,exp_tas,exp_das',
    [
        (
            np.linspace(0, 1, 11), np.linspace(0, 1, 11), 1, 0,
            4, np.array([4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1])
        ),
        (
            np.linspace(0, 1, 11), np.linspace(0, 1, 11), 0, 0,
            3, np.array([3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0])
        )
    ]
)
def test_recall_anomalies_curve(
    y_score: np.ndarray,
    thresholds: T.List[float],
    pos_label: int,
    k: float,
    exp_tas: int,
    exp_das: np.ndarray
):
    y = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1])
    exp_tas = np.ones(len(y)) * exp_tas

    das, tas = _recall_anomalies_curve(
        y,
        y_score,
        thresholds=thresholds,
        pos_label=pos_label,
        k=k
    )

    np.testing.assert_array_equal(exp_tas, tas)
    np.testing.assert_array_equal(exp_das, das)
