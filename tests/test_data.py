import os
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from testfixtures import TempDirectory

import mlnext


class TestLoadData(TestCase):
    def setUp(self):
        self.d = TempDirectory()

        self.df = pd.DataFrame([0, 1, 2], columns=['a'])
        self.df_path = os.path.join(self.d.path, 'data.csv')
        self.df.to_csv(self.df_path, index=False)

    def tearDown(self):
        self.d.cleanup()

    def test_load_data(self):
        result = mlnext.load_data(path=self.df_path)

        pd.testing.assert_frame_equal(result, self.df)


class TestTemporalize(TestCase):

    def test_temporalize(self):
        arr = np.arange(12).reshape(-1, 2)
        expected = np.arange(12).reshape(-1, 2, 2)

        result = mlnext.temporalize(data=arr, timesteps=2)

        np.testing.assert_array_equal(result, expected)

    def test_temporalize_drop_rows(self):
        arr = np.arange(14).reshape(-1, 2)
        expected = np.arange(12).reshape(-1, 3, 2)

        result = mlnext.temporalize(data=arr, timesteps=3)

        np.testing.assert_array_equal(result, expected)


class TestLoadData3D(TestCase):
    def setUp(self):
        self.d = TempDirectory()

        self.arr = np.arange(8).reshape(4, 2)
        self.df = pd.DataFrame(self.arr, columns=['a', 'b'])
        self.df_path = os.path.join(self.d.path, 'data.csv')
        self.df.to_csv(self.df_path, index=False)

    def tearDown(self):
        self.d.cleanup()

    def test_load_data_3d(self):
        result = mlnext.load_data_3d(path=self.df_path, timesteps=2)
        expected = self.arr.reshape(2, 2, 2)

        np.testing.assert_array_equal(result, expected)


class TestDetemporalize(TestCase):

    def test_detemporalize(self):

        arr = np.arange(12).reshape(3, 2, 2)
        expected = np.arange(12).reshape(6, 2)

        result = mlnext.detemporalize(data=arr)

        np.testing.assert_array_equal(result, expected)

    def test_detemporalize_2d(self):

        arr = np.arange(8).reshape(4, 2)

        result = mlnext.detemporalize(data=arr)

        np.testing.assert_array_equal(result, arr)


class TestSample(TestCase):

    def test_sample_normal(self):

        mean = 0
        std = 1

        np.random.seed(0)
        expected = np.random.normal(loc=mean, scale=std)

        np.random.seed(0)
        result = mlnext.sample_normal(mean=mean, std=std)

        np.testing.assert_array_equal(result, expected)

    def test_sample_bernoulli(self):

        mean = 0.2
        np.random.seed(0)
        expected = np.random.binomial(n=1, p=mean)

        np.random.seed(0)
        result = mlnext.sample_bernoulli(mean=mean)

        self.assertEqual(result, expected)


@pytest.fixture
def data(request) -> np.ndarray:
    """Create a numpy array. Expects request to be a tuple of (rows, features).

    Args:
        request (Any): Tuple of (rows, features).

    Returns:
        np.ndarray: Returns an numpy array of shape (rows, features).
    """
    rows, features = request.param
    i, j = np.ogrid[:rows, :features]
    return 10 * i + j


@pytest.mark.parametrize(
    'data,timesteps,stride,exp',
    [
        ((6, 3), 2, 0,
         np.array([[[0, 1, 2],
                    [10, 11, 12]],

                   [[20, 21, 22],
                    [30, 31, 32]],

                   [[40, 41, 42],
                    [50, 51, 52]]])),
        ((6, 3), 2, 2,
         np.array([[[0, 1, 2],
                    [10, 11, 12]],

                   [[20, 21, 22],
                    [30, 31, 32]],

                   [[40, 41, 42],
                    [50, 51, 52]]])),
        ((6, 3), 3, 1,
         np.array([[[0, 1, 2],
                    [10, 11, 12],
                    [20, 21, 22]],

                   [[10, 11, 12],
                    [20, 21, 22],
                    [30, 31, 32]],

                   [[20, 21, 22],
                    [30, 31, 32],
                    [40, 41, 42]],

                   [[30, 31, 32],
                    [40, 41, 42],
                    [50, 51, 52]]])),
        ((8, 3), 3, 2,
         np.array([[[0, 1, 2],
                    [10, 11, 12],
                    [20, 21, 22]],

                   [[20, 21, 22],
                    [30, 31, 32],
                    [40, 41, 42]],

                   [[40, 41, 42],
                    [50, 51, 52],
                    [60, 61, 62]]]))
    ],
    indirect=['data']
)
def test_temporalize(
    data: np.ndarray,
    timesteps: int,
    stride: int,
    exp: np.ndarray
):

    result = mlnext.temporalize(data, timesteps=timesteps, stride=stride)
    print(result)
    np.testing.assert_array_equal(result, exp)


@pytest.mark.parametrize(
    'data,timesteps,stride,err_msg',
    [
        (np.ones(10), 10, 10,
         'Expected array of dimension 2, but got 1 for array at position 0.'),
        (np.ones((10, 1)), 10, -2,
         'Stride must be greater than 0.'),
        (np.ones((10, 1)), -1, 1,
         'Timesteps must be greater than 1.')
    ]
)
def test_temporalize_fails(
    data: np.ndarray,
    timesteps: int,
    stride: int,
    err_msg: str
):
    with pytest.raises(ValueError) as exc_info:
        mlnext.temporalize(data, timesteps=timesteps, stride=stride)

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'data,timesteps,stride,last_point_only,exp',
    [
        ((6, 3), 2, 0, False,
         np.array([[0, 1, 2],
                   [10, 11, 12],
                   [20, 21, 22],
                   [30, 31, 32],
                   [40, 41, 42],
                   [50, 51, 52]])
         ),
        ((6, 3), 3, 1, False,
         np.array([[0, 1, 2],
                   [10, 11, 12],
                   [20, 21, 22],
                   [30, 31, 32],
                   [40, 41, 42],
                   [50, 51, 52]])
         ),
        ((6, 3), 3, 2, False,
         np.array([[0, 1, 2],
                   [10, 11, 12],
                   [20, 21, 22],
                   [30, 31, 32],
                   [40, 41, 42]])
         ),
        ((6, 3), 3, 1, True,
         np.array([[20, 21, 22],
                   [30, 31, 32],
                   [40, 41, 42],
                   [50, 51, 52]])
         )
    ],
    indirect=['data']
)
def test_detemporalize(
    data: np.ndarray,
    timesteps: int,
    stride: int,
    last_point_only: bool,
    exp: np.ndarray
):
    data_3d = mlnext.temporalize(data, timesteps=timesteps, stride=stride)
    result = mlnext.detemporalize(
        data_3d, stride=stride, last_point_only=last_point_only)

    np.testing.assert_array_equal(result, exp)
