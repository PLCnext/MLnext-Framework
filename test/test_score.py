from unittest import TestCase

import numpy as np
import pytest
import scipy.stats

from mlnext import score
from mlnext.score import kl_divergence


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


class TestThreshold(TestCase):

    def test_get_threshold_100(self):

        data = np.array([0, 1.])

        expected = 1.
        result = score.get_threshold(x=data)

        self.assertEqual(result, expected)

    def test_get_threshold_50(self):

        data = np.array([0, 1.])

        expected = 0.5
        result = score.get_threshold(x=data, p=50)

        self.assertEqual(result, expected)

    def test_get_threshold_80(self):

        data = np.array([0, 1.])

        expected = 0.8
        result = score.get_threshold(x=data, p=80)

        self.assertEqual(result, expected)

    def test_apply_threshold(self):

        data = np.array([0, 0.4, 0.6, 1.])
        expected = np.array([0, 0, 1, 1])

        result = score.apply_threshold(data, threshold=0.5)

        np.testing.assert_array_equal(result, expected)


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

    def test_eval_sigmoid(self):

        data = np.array([[0.2], [0.6]])
        expected = np.array([[0], [1]])

        result = score.eval_sigmoid(y=data)

        np.testing.assert_array_equal(result, expected)

    def test_eval_sigmoid_3d(self):

        data = np.array([[[0.2], [0.6]]])
        expected = np.array([[0], [1]])

        result = score.eval_sigmoid(y=data)

        np.testing.assert_array_equal(result, expected)

    def test_eval_sigmoid_invert(self):

        data = np.array([[0.2], [0.6]])
        expected = np.array([[1], [0]])

        result = score.eval_sigmoid(y=data, invert=True)

        np.testing.assert_array_equal(result, expected)

    def test_eval_sigmoid_3d_invert(self):

        data = np.array([[[0.2], [0.6]]])
        expected = np.array([[1], [0]])

        result = score.eval_sigmoid(y=data, invert=True)

        np.testing.assert_array_equal(result, expected)


class TestMovingAverage(TestCase):

    def test_moving_average(self):

        data = np.arange(5)
        expected = [0, 0.5, 1.5, 2.5, 3.5, 2]
        result = score.moving_average(data, 2, mode='full')

        np.testing.assert_array_equal(result, expected)


class TestMetrics(TestCase):

    def test_eval_metrics(self):

        y = np.ones((10, 1))
        y_hat = np.ones((10, 1))

        expected = {'accuracy': 1.0, 'precision': 1.0,
                    'recall': 1.0, 'f1': 1.0}
        result = score.eval_metrics(y, y_hat)

        self.assertDictEqual(result, expected)

    def test_eval_metrics_uneven_length(self):

        y = np.ones((10, 1))
        y_hat = np.ones((12, 1))

        expected = {'accuracy': 1.0, 'precision': 1.0,
                    'recall': 1.0, 'f1': 1.0}
        result = score.eval_metrics(y, y_hat)

        self.assertDictEqual(result, expected)

    def test_eval_metrics_all(self):

        y = [np.ones((10, 1)), np.zeros((10, 1))]
        y_hat = [np.ones((10, 1)), np.zeros((10, 1))]

        expected = {'accuracy': 1.0, 'precision': 1.0,
                    'recall': 1.0, 'f1': 1.0, 'AUC': 1.0}
        result = score.eval_metrics_all(y, y_hat)

        self.assertDictEqual(result, expected)


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
    mean: np.array,
    log_var: np.array,
    prior_mean: float,
    prior_std: float,
    exp: np.array
):

    np.testing.assert_array_almost_equal(kl_divergence(
        mean=mean,
        log_var=log_var,
        prior_mean=prior_mean,
        prior_std=prior_std
    ), exp)
