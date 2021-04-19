from unittest import TestCase

import numpy as np
import pandas as pd

from mlutils import pipeline


class TestColumnSelector(TestCase):

    def setUp(self):

        data = np.arange(8).reshape(-1, 2)
        cols = ['a', 'b']
        self.df = pd.DataFrame(data, columns=cols)

    def test_select_columns(self):
        t = pipeline.ColumnSelector(keys=['a'])

        expected = self.df.loc[:, ['a']]
        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestColumnDropper(TestCase):

    def setUp(self):

        data = np.arange(8).reshape(-1, 2)
        cols = ['a', 'b']
        self.df = pd.DataFrame(data, columns=cols)

    def test_drop_columns(self):

        t = pipeline.ColumnDropper(columns=['b'])

        expected = self.df.loc[:, ['a']]
        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_drop_columns_verbose(self):

        t = pipeline.ColumnDropper(columns=['b'], verbose=True)

        expected = self.df.loc[:, ['a']]
        result = t.transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_drop__missing_columns(self):

        t = pipeline.ColumnDropper(columns=['c'])

        with self.assertWarns(Warning):
            t.transform(self.df)


class TestColumnRename(TestCase):
    pass
