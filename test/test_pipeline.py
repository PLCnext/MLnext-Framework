import datetime
import typing as T
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from mlnext import pipeline


class TestColumnSelector(TestCase):

    def setUp(self):

        data = np.arange(8).reshape(-1, 2)
        cols = ['a', 'b']
        self.df = pd.DataFrame(data, columns=cols)

    def test_select_columns(self):
        t = pipeline.ColumnSelector(keys=['a'])

        expected = self.df.loc[:, ['a']]
        result = t.fit_transform(self.df.copy())

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

    def test_rename_columns(self):

        t = pipeline.ColumnRename(lambda x: x.split('.')[-1])
        df = pd.DataFrame(columns=['a.b.c', 'd.e.f'])
        expected = pd.DataFrame(columns=['c', 'f'])

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestNaDropper(TestCase):

    def test_drop_na(self):

        t = pipeline.NaDropper()
        df = pd.DataFrame([1, 0, pd.NA])
        expected = pd.DataFrame([1, 0], dtype=object)

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestClip(TestCase):

    def test_clip(self):

        t = pipeline.Clip(lower=0.5, upper=1.5)
        df = pd.DataFrame([[0.1, 0.4, 0.6, 0.8, 1.2, 1.5]])
        expected = pd.DataFrame([[0.5, 0.5, 0.6, 0.8, 1.2, 1.5]])

        result = t.fit_transform(df)
        pd.testing.assert_frame_equal(result, expected)


class TestDatetimeTransformer(TestCase):
    # FIXME: fails in gitlab pipeline but succeeds locally
    def test_datetime(self):

        t = pipeline.DatetimeTransformer(columns=['time'])
        df = pd.DataFrame([['2021-01-04 14:12:31']], columns=['time'])
        expected = pd.DataFrame([[datetime.datetime(2021, 1, 4, 14, 12, 31)]],
                                columns=['time'])

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_datetime_missing_cols(self):

        t = pipeline.DatetimeTransformer(columns=['t'])
        df = pd.DataFrame([['2021-01-04 14:12:31']], columns=['time'])

        with self.assertRaises(ValueError):
            t.fit_transform(df)


class TestNumericTransformer(TestCase):
    # FIXME: fails in gitlab pipeline but succeeds locally
    def test_numeric(self):

        t = pipeline.NumericTransformer(columns=['1'])
        df = pd.DataFrame([0, 1], columns=['1'], dtype=object)
        expected = pd.DataFrame([0, 1], columns=['1'], dtype=np.int64)

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_numeric_missing_column(self):

        t = pipeline.NumericTransformer(columns=['2'])
        df = pd.DataFrame([0, 1], columns=['1'], dtype=object)

        with self.assertRaises(ValueError):
            t.fit_transform(df)

    def test_numeric_additional_column(self):

        t = pipeline.NumericTransformer(columns=['2'])
        df = pd.DataFrame([[0, 1]], columns=['1', '2'], dtype=object)
        expected = pd.DataFrame([[0, 1]], columns=['1', '2'], dtype=object)
        expected['2'] = expected['2'].apply(pd.to_numeric)
        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_numeric_multiple_column(self):

        t = pipeline.NumericTransformer(columns=['1', '2'])
        df = pd.DataFrame([[0, 1]], columns=['1', '2'], dtype=object)
        expected = pd.DataFrame([[0, 1]], columns=['1', '2'])

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_numeric_all_column(self):

        t = pipeline.NumericTransformer()
        df = pd.DataFrame([[0, 1]], columns=['1', '2'], dtype=object)
        expected = pd.DataFrame([[0, 1]], columns=['1', '2'])

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestTimeframeExtractor(TestCase):
    def setUp(self):

        self.dates = [datetime.datetime(2021, 10, 1, 9, 50, 0),
                      datetime.datetime(2021, 10, 1, 10, 0, 0),
                      datetime.datetime(2021, 10, 1, 11, 0, 0),
                      datetime.datetime(2021, 10, 1, 12, 0, 0),
                      datetime.datetime(2021, 10, 1, 12, 10, 0)]
        self.values = np.arange(len(self.dates))
        self.df = pd.DataFrame(zip(self.dates, self.values),
                               columns=['time', 'value'])

    def test_timeframe_extractor(self):

        t = pipeline.TimeframeExtractor(
            time_column='time', start_time=datetime.time(10, 0, 0),
            end_time=datetime.time(12, 0, 0), verbose=True)
        expected = pd.DataFrame(zip(self.dates[1:-1], np.arange(1, 4)),
                                columns=['time', 'value'])

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_timeframe_extractor_invert(self):

        t = pipeline.TimeframeExtractor(
            time_column='time', start_time=datetime.time(10, 0, 0),
            end_time=datetime.time(12, 0, 0), invert=True)
        expected = pd.DataFrame(zip([self.dates[0], self.dates[-1]],
                                    np.array([0, 4])),
                                columns=['time', 'value'])

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestDateExtractor(TestCase):

    def setUp(self):
        self.dates = [datetime.datetime(2021, 10, 1, 9, 50, 0),
                      datetime.datetime(2021, 10, 2, 10, 0, 0),
                      datetime.datetime(2021, 10, 3, 11, 0, 0),
                      datetime.datetime(2021, 10, 4, 12, 0, 0),
                      datetime.datetime(2021, 10, 5, 12, 10, 0)]
        self.values = np.arange(len(self.dates))
        self.df = pd.DataFrame(zip(self.dates, self.values),
                               columns=['date', 'value'])

    def test_date_extractor(self):

        t = pipeline.DateExtractor(
            date_column='date', start_date=datetime.date(2021, 10, 2),
            end_date=datetime.date(2021, 10, 4), verbose=True)
        expected = pd.DataFrame(zip(self.dates[1:-1], np.arange(1, 4)),
                                columns=['date', 'value'])

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_date_extractor_invert(self):

        t = pipeline.DateExtractor(
            date_column='date', start_date=datetime.date(2021, 10, 2),
            end_date=datetime.date(2021, 10, 4), invert=True)
        expected = pd.DataFrame(zip([self.dates[0], self.dates[-1]],
                                    np.array([0, 4])),
                                columns=['date', 'value'])

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestValueMapper(TestCase):

    def test_value_mapper_one_column(self):

        t = pipeline.ValueMapper(columns=['b'], classes={2.0: 1.0})
        df = pd.DataFrame(np.ones((3, 2)) * 2, columns=['a', 'b'])
        expected = pd.DataFrame(zip(np.ones((3, 1)) * 2, np.ones((3, 1))),
                                columns=['a', 'b'], dtype=np.float64)

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_value_mapper_all_columns(self):

        t = pipeline.ValueMapper(columns=['a', 'b'], classes={2.0: 1.0})
        df = pd.DataFrame(np.ones((3, 2)) * 2, columns=['a', 'b'])
        expected = pd.DataFrame(np.ones((3, 2)), columns=['a', 'b'],
                                dtype=np.float64)

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_value_mapper_missing_value(self):

        t = pipeline.ValueMapper(columns=['a', 'b'], classes={2.0: 1.0})
        df = pd.DataFrame(np.ones((3, 2)), columns=['a', 'b'])
        expected = pd.DataFrame(np.ones((3, 2)), columns=['a', 'b'],
                                dtype=np.float64)

        with self.assertWarns(Warning):
            result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestSorter(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'a': [2, 3, 1, 4],
            'b': ['A', 'D', 'C', 'B']
        })

    def test_sorter(self):

        t = pipeline.Sorter(columns=['a'])
        result = t.fit_transform(self.df)
        expected = self.df.copy().sort_values(by=['a'])

        pd.testing.assert_frame_equal(result, expected)

    def test_sorter_multi_col(self):

        t = pipeline.Sorter(columns=['a', 'b'])
        result = t.fit_transform(self.df)
        expected = self.df.copy().sort_values(by=['a', 'b'])

        pd.testing.assert_frame_equal(result, expected)


class TestFill(TestCase):

    def setUp(self):
        self.df = pd.DataFrame([[0.0, 1.0, 0.2, pd.NA, 0.5]])

    def test_fill(self):

        t = pipeline.Fill(value=1.0)
        expected = pd.DataFrame([[0.0, 1.0, 0.2, 1.0, 0.5]])

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestTimeOffsetTransformer(TestCase):

    def test_timeoffset(self):

        t = pipeline.TimeOffsetTransformer(
            time_columns=['t'], timedelta=pd.Timedelta(1, 'h'))
        df = pd.DataFrame({'t': [datetime.datetime(2020, 10, 1, 12, 3, 10)]})
        expected = pd.DataFrame(
            {'t': [datetime.datetime(2020, 10, 1, 13, 3, 10)]})

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_timeoffset_multi_col(self):

        t = pipeline.TimeOffsetTransformer(
            time_columns=['t'], timedelta=pd.Timedelta(1, 'h'))
        df = pd.DataFrame({'t': [datetime.datetime(2020, 10, 1, 12, 3, 10)],
                           'tt': [datetime.datetime(2020, 10, 1, 13, 3, 10)]})
        expected = pd.DataFrame(
            {'t': [datetime.datetime(2020, 10, 1, 13, 3, 10)],
             'tt': [datetime.datetime(2020, 10, 1, 13, 3, 10)]})

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestConditionedDropper(TestCase):

    def setUp(self):
        self.data = [0.0, 0.5, 1.0, 1.2]
        self.df = pd.DataFrame({'a': self.data})

    def test_dropper(self):
        t = pipeline.ConditionedDropper(column='a', threshold=1.0)
        expected = pd.DataFrame({'a': self.data[:-1]})

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_dropper_invert(self):
        t = pipeline.ConditionedDropper(column='a', threshold=1.0, invert=True)
        expected = pd.DataFrame({'a': self.data[-2:]})

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestZeroVarianceDropper(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'one': np.ones((4,)),
                                'zeros': np.zeros((4,)),
                                'mixed': np.arange(4)})

    def test_dropper(self):
        t = pipeline.ZeroVarianceDropper(verbose=True)
        expected = pd.DataFrame({
            'mixed': np.arange(4)
        })

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_dropper_fit_higher_variance(self):

        t = pipeline.ZeroVarianceDropper()
        expected = pd.DataFrame({
            'mixed': np.arange(4)
        })

        t.fit(self.df)

        df = self.df.copy()
        df.iloc[0, 0] = 0
        with self.assertWarns(Warning):
            result = t.transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestSignalSorter(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'one': np.ones((4,)),
                                'zeros': np.zeros((4,)),
                                'mixed': np.arange(4),
                                'binary': [0, 1, 0, 1],
                                'cont': [10, 11, 10, 11]})

    def test_sorter(self):
        t = pipeline.SignalSorter(verbose=True)
        expected = self.df.loc[:, ['mixed', 'cont', 'one', 'zeros', 'binary']]

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestColumnSorter(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'one': np.ones((2,)),
                                'zeros': np.zeros((2,)),
                                'mixed': np.arange(2),
                                'binary': [0, 1]})

        self.df2 = pd.DataFrame({'mixed': np.arange(2),
                                 'zeros': np.zeros((2,)),
                                 'one': np.ones((2,)),
                                 'binary': [0, 1]})

    def test_sorter(self):
        t = pipeline.ColumnSorter(verbose=True)

        t.fit_transform(self.df)
        result = t.transform(self.df2)

        pd.testing.assert_frame_equal(result, self.df)

    def test_sorter_missing_columns_raise(self):
        t = pipeline.ColumnSorter()

        t.fit_transform(self.df)

        with self.assertRaises(ValueError):
            t.transform(self.df2.iloc[:, :3])

    def test_sorter_additional_columns_raise(self):
        t = pipeline.ColumnSorter()

        t.fit_transform(self.df)

        df = self.df2.copy()
        df['T'] = [0, 1]
        with self.assertRaises(ValueError):
            t.transform(df)

    def test_sorter_additional_columns_warn(self):
        t = pipeline.ColumnSorter(raise_on_error=False)

        t.fit_transform(self.df)

        df = self.df2.copy()
        df['T'] = [0, 1]

        with self.assertWarns(Warning):
            result = t.transform(df)

        pd.testing.assert_frame_equal(result, self.df)


class TestDifferentialCreator(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'test': [0.0, 1.0, 0.2, 0.3, 0.5]})

        self.df_sel = pd.DataFrame({'test': [0.0, 1.0, 0.2, 0.3, 0.5],
                                    'test2': [0.0, 1.0, 0.2, 0.3, 0.5]})

    def test_creator(self):
        t = pipeline.DifferentialCreator(columns=['test'])

        expected = pd.DataFrame({'test': [0.0, 1.0, 0.2, 0.3, 0.5],
                                 'test_dif': [0.0, 1.0, -0.8, 0.1, 0.2]})

        result = t.fit_transform(self.df)
        pd.testing.assert_frame_equal(result, expected)

    def test_creator_selection(self):
        t = pipeline.DifferentialCreator(columns=['test'])

        expected = pd.DataFrame({'test': [0.0, 1.0, 0.2, 0.3, 0.5],
                                 'test2': [0.0, 1.0, 0.2, 0.3, 0.5],
                                 'test_dif': [0.0, 1.0, -0.8, 0.1, 0.2]})

        result = t.fit_transform(self.df_sel)
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'feature_range,clip,p,exp',
    [
        (
            (0, 1.), None, 100,
            pd.DataFrame({
                'a': [-3., -2., -1., 0., 1., 2., 3.],
                'b': [-3., -2., -1., 0., 1., 2., 3.]
            })
        ),
        (
            (0, .5), None, 100,
            pd.DataFrame({
                'a': [-1.5, -1., -.5, 0., .5, 1., 1.5],
                'b': [-1.5, -1., -.5, 0., .5, 1., 1.5]
            })
        ),
        (
            (0, .5), (-1, 1.), 100,
            pd.DataFrame({
                'a': [-1., -1., -.5, 0., .5, 1., 1.],
                'b': [-1., -1., -.5, 0., .5, 1., 1.]
            })
        ),
        (
            (0, .5), (0, 1.), 100,
            pd.DataFrame({
                'a': [0., 0., 0., 0., .5, 1., 1.],
                'b': [0., 0., 0., 0., .5, 1., 1.]
            })
        )
    ]
)
def test_ClippingMinMaxScaler(
    feature_range: T.Tuple[float, float],
    clip: T.Optional[T.Tuple[float, float]],
    p: float,
    exp: pd.DataFrame,
):

    scaler = pipeline.ClippingMinMaxScaler(
        feature_range,
        clip=clip,
        p=p
    )

    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [0, 1, 2, 3]})
    scaler.fit_transform(df)

    df = pd.DataFrame({
        'a': [-8, -5, -2, 1, 4, 7, 10],
        'b': [-9, -6, -3, 0, 3, 6, 9]
    })
    result = scaler.transform(df)
    pd.testing.assert_frame_equal(result, exp)
