import os
from unittest import TestCase

import numpy as np
import pandas as pd
from testfixtures import TempDirectory

from mlnext import data


class TestLoadData(TestCase):
    def setUp(self):
        self.d = TempDirectory()

        self.df = pd.DataFrame([0, 1, 2], columns=['a'])
        self.df_path = os.path.join(self.d.path, 'data.csv')
        self.df.to_csv(self.df_path, index=False)

    def tearDown(self):
        self.d.cleanup()

    def test_load_data(self):
        result = data.load_data(path=self.df_path)

        pd.testing.assert_frame_equal(result, self.df)


class TestTemporalize(TestCase):

    def test_temporalize(self):
        arr = np.arange(12).reshape(-1, 2)
        expected = np.arange(12).reshape(-1, 2, 2)

        result = data.temporalize(data=arr, timesteps=2)

        np.testing.assert_array_equal(result, expected)

    def test_temporalize_drop_rows(self):
        arr = np.arange(14).reshape(-1, 2)
        expected = np.arange(12).reshape(-1, 3, 2)

        result = data.temporalize(data=arr, timesteps=3)

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
        result = data.load_data_3d(path=self.df_path, timesteps=2)
        expected = self.arr.reshape(2, 2, 2)

        np.testing.assert_array_equal(result, expected)


class TestDetemporalize(TestCase):

    def test_detemporalize(self):

        arr = np.arange(8).reshape(2, 2, 2)
        expected = np.arange(8).reshape(4, 2)

        result = data.detemporalize(data=arr)

        np.testing.assert_array_equal(result, expected)

    def test_detemporalize_2d(self):

        arr = np.arange(8).reshape(4, 2)

        result = data.detemporalize(data=arr)

        np.testing.assert_array_equal(result, arr)
