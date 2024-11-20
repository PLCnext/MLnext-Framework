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

    def test_select_columns_deprecated(self):
        t = pipeline.ColumnSelector(keys=['a'])

        expected = self.df.loc[:, ['a']]
        result = t.fit_transform(self.df.copy())

        pd.testing.assert_frame_equal(result, expected)

    def test_select_columns(self):
        t = pipeline.ColumnSelector(columns=['a'])

        expected = self.df.loc[:, ['a']]
        result = t.fit_transform(self.df.copy())

        pd.testing.assert_frame_equal(result, expected)

    def test_fit_columns(self):
        t = pipeline.ColumnSelector()

        expected = self.df.loc[:, ['a']]
        t.fit(pd.DataFrame({'a': [0]}))
        result = t.transform(self.df)

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
    def test_rename_columns_lambda(self):
        t = pipeline.ColumnRename(lambda x: x.split('.')[-1])
        df = pd.DataFrame(columns=['a.b.c', 'd.e.f'])
        expected = pd.DataFrame(columns=['c', 'f'])

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_rename_columns_dict(self):
        t = pipeline.ColumnRename({'a.b.c': 'g.h.i'})
        df = pd.DataFrame(columns=['a.b.c', 'd.e.f'])
        expected = pd.DataFrame(columns=['g.h.i', 'd.e.f'])

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


@pytest.mark.parametrize(
    'columns,lower,upper,exp',
    [
        (
            None,
            0.0,
            0.5,
            pd.DataFrame({'a': [0.0, 0.23, 0.5], 'b': [0.25, 0.5, 0.5]}),
        ),
        (
            ['a'],
            0.0,
            0.5,
            pd.DataFrame({'a': [0.0, 0.23, 0.5], 'b': [0.25, 1.24, 0.68]}),
        ),
        (
            ['b'],
            0.0,
            0.5,
            pd.DataFrame({'a': [-0.56, 0.23, 0.67], 'b': [0.25, 0.5, 0.5]}),
        ),
    ],
)
def test_clip(
    columns: T.List[str], lower: float, upper: float, exp: pd.DataFrame
):
    df = pd.DataFrame({'a': [-0.56, 0.23, 0.67], 'b': [0.25, 1.24, 0.68]})
    t = pipeline.Clip(columns=columns, upper=upper, lower=lower)

    result = t.fit_transform(df)

    pd.testing.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    'columns,lower,upper,msg',
    [
        (
            ['c'],
            0.0,
            0.5,
            "Columns ['c'] not found in DataFrame with columns ['a', 'b'].",
        ),
    ],
)
def test_clip_raises(
    columns: T.List[str], lower: float, upper: float, msg: str
):
    df = pd.DataFrame({'a': [-0.56, 0.23, 0.67], 'b': [0.25, 1.24, 0.68]})
    t = pipeline.Clip(columns=columns, upper=upper, lower=lower)

    with pytest.raises(ValueError) as exc_info:
        t.fit_transform(df)

    assert exc_info.value.args[0] == msg


@pytest.mark.parametrize(
    'columns,dt_format,exp',
    [
        (
            ['time'],
            None,
            pd.DataFrame(
                [[datetime.datetime(2021, 1, 4, 14, 12, 31)]], columns=['time']
            ),
        ),
        (
            ['time'],
            '%Y-%m-%d %H:%M:%S',
            pd.DataFrame(
                [[datetime.datetime(2021, 1, 4, 14, 12, 31)]], columns=['time']
            ),
        ),
    ],
)
def test_datetime_transformer(
    columns: T.List[str], dt_format: T.Optional[str], exp: pd.DataFrame
):
    df = pd.DataFrame([['2021-01-04 14:12:31']], columns=['time'])
    t = pipeline.DatetimeTransformer(columns=columns, dt_format=dt_format)

    result = t.fit_transform(df)

    pd.testing.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    'columns,dt_format,msg',
    [
        (
            ['t'],
            None,
            "Columns ['t'] not found in DataFrame with columns ['time'].",
        )
    ],
)
def test_datetime_transformer_raises(
    columns: T.List[str], dt_format: T.Optional[str], msg: str
):
    df = pd.DataFrame([['2021-01-04 14:12:31']], columns=['time'])
    t = pipeline.DatetimeTransformer(columns=columns, dt_format=dt_format)

    with pytest.raises(ValueError) as exc_info:
        t.fit_transform(df)

    assert exc_info.value.args[0] == msg


@pytest.mark.parametrize(
    'columns,exp',
    [
        (['a'], pd.DataFrame({'a': [0, 1, 2], 'b': ['3', '4', 0]})),
        (['a', 'b'], pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 0]})),
        (None, pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 0]})),
    ],
)
def test_numeric_transformer(
    columns: T.Optional[T.List[str]], exp: pd.DataFrame
):
    df = pd.DataFrame({'a': [0, '1', 2], 'b': ['3', '4', 0]})
    t = pipeline.NumericTransformer(columns=columns)

    result = t.fit_transform(df)

    pd.testing.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    'columns,msg',
    [
        (
            ['c'],
            "Columns ['c'] not found in DataFrame with columns ['a', 'b'].",
        ),
    ],
)
def test_numeric_transformer_raises(
    columns: T.Optional[T.List[str]], msg: str
):
    df = pd.DataFrame({'a': [0, '1', 2], 'b': ['3', '4', 0]})
    t = pipeline.NumericTransformer(columns=columns)

    with pytest.raises(ValueError) as exc_info:
        t.fit_transform(df)

    assert exc_info.value.args[0] == msg


@pytest.mark.parametrize(
    'time_column,start_time,end_time,invert,exp',
    [
        ('time', '10:00:00', '12:00:00', False, [1, 2, 3]),
        (
            'time',
            datetime.time(10, 0, 0),
            datetime.time(12, 0, 0),
            False,
            [1, 2, 3],
        ),
        ('time', '10:00:00', '12:00:00', True, [0, 4]),
        (
            'time',
            datetime.time(10, 0, 0),
            datetime.time(12, 0, 0),
            True,
            [0, 4],
        ),
    ],
)
def test_timeframe_extractor(
    time_column: str,
    start_time: T.Union[str, datetime.time],
    end_time: T.Union[str, datetime.time],
    invert: bool,
    exp: T.List[int],
):
    dates = [
        datetime.datetime(2021, 10, 1, 9, 50, 0),
        datetime.datetime(2021, 10, 1, 10, 0, 0),
        datetime.datetime(2021, 10, 1, 11, 0, 0),
        datetime.datetime(2021, 10, 1, 12, 0, 0),
        datetime.datetime(2021, 10, 1, 12, 10, 0),
    ]
    df = pd.DataFrame(
        zip(dates, np.arange(len(dates))), columns=['time', 'value']
    )
    t = pipeline.TimeframeExtractor(
        time_column=time_column,
        start_time=start_time,
        end_time=end_time,
        invert=invert,
    )

    result = t.fit_transform(df)

    pd.testing.assert_frame_equal(result, df.loc[exp].reset_index(drop=True))


@pytest.mark.parametrize(
    'date_column,start_date,end_date,invert,exp',
    [
        ('date', '2021-10-02', '2021-10-04', False, [1, 2, 3]),
        (
            'date',
            datetime.date(2021, 10, 2),
            datetime.date(2021, 10, 4),
            False,
            [1, 2, 3],
        ),
        ('date', '2021-10-02', '2021-10-04', True, [0, 4]),
        (
            'date',
            datetime.date(2021, 10, 2),
            datetime.date(2021, 10, 4),
            True,
            [0, 4],
        ),
    ],
)
def test_date_extractor(
    date_column: str,
    start_date: T.Union[str, datetime.date],
    end_date: T.Union[str, datetime.date],
    invert: bool,
    exp: T.List[int],
):
    dates = [
        datetime.datetime(2021, 10, 1, 9, 50, 0),
        datetime.datetime(2021, 10, 2, 10, 0, 0),
        datetime.datetime(2021, 10, 3, 11, 0, 0),
        datetime.datetime(2021, 10, 4, 12, 0, 0),
        datetime.datetime(2021, 10, 5, 12, 10, 0),
    ]
    df = pd.DataFrame(
        zip(dates, np.arange(len(dates))), columns=['date', 'value']
    )
    t = pipeline.DateExtractor(
        date_column=date_column,
        start_date=start_date,
        end_date=end_date,
        invert=invert,
    )

    result = t.fit_transform(df)

    pd.testing.assert_frame_equal(result, df.loc[exp].reset_index(drop=True))


class TestValueMapper(TestCase):
    def test_value_mapper_one_column(self):
        t = pipeline.ValueMapper(columns=['b'], classes={2.0: 1.0})
        df = pd.DataFrame(np.ones((3, 2)) * 2, columns=['a', 'b'])
        expected = pd.DataFrame(
            zip(np.ones((3, 1)) * 2, np.ones((3, 1))),
            columns=['a', 'b'],
            dtype=np.float64,
        )

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_value_mapper_all_columns(self):
        t = pipeline.ValueMapper(columns=['a', 'b'], classes={2.0: 1.0})
        df = pd.DataFrame(np.ones((3, 2)) * 2, columns=['a', 'b'])
        expected = pd.DataFrame(
            np.ones((3, 2)), columns=['a', 'b'], dtype=np.float64
        )

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_value_mapper_missing_value(self):
        t = pipeline.ValueMapper(columns=['a', 'b'], classes={2.0: 1.0})
        df = pd.DataFrame(np.ones((3, 2)), columns=['a', 'b'])
        expected = pd.DataFrame(
            np.ones((3, 2)), columns=['a', 'b'], dtype=np.float64
        )

        with self.assertWarns(Warning):
            result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestSorter(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': [2, 3, 1, 4], 'b': ['A', 'D', 'C', 'B']})

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


@pytest.mark.parametrize(
    'value,method,exp',
    [
        (0.0, None, [0.0, 1.0, 0.2, 0.0, 0.0, 0.5]),
        (1.0, None, [0.0, 1.0, 0.2, 1.0, 1.0, 0.5]),
        (None, 'bfill', [0.0, 1.0, 0.2, 0.5, 0.5, 0.5]),
        (None, 'ffill', [0.0, 1.0, 0.2, 0.2, 0.2, 0.5]),
    ],
)
def test_fill(value: T.Optional[T.Any], method: T.Optional[str], exp: T.List):
    df = pd.DataFrame([0.0, 1.0, 0.2, np.nan, np.nan, 0.5])
    scaler = pipeline.Fill(value=value, method=method)

    result = scaler.fit_transform(df)

    pd.testing.assert_frame_equal(result, pd.DataFrame(exp))


class TestTimeOffsetTransformer(TestCase):
    def test_timeoffset(self):
        t = pipeline.TimeOffsetTransformer(
            time_columns=['t'], timedelta=pd.Timedelta(1, 'h')
        )
        df = pd.DataFrame({'t': [datetime.datetime(2020, 10, 1, 12, 3, 10)]})
        expected = pd.DataFrame(
            {'t': [datetime.datetime(2020, 10, 1, 13, 3, 10)]}
        )

        result = t.fit_transform(df)

        pd.testing.assert_frame_equal(result, expected)

    def test_timeoffset_multi_col(self):
        t = pipeline.TimeOffsetTransformer(
            time_columns=['t'], timedelta=pd.Timedelta(1, 'h')
        )
        df = pd.DataFrame(
            {
                't': [datetime.datetime(2020, 10, 1, 12, 3, 10)],
                'tt': [datetime.datetime(2020, 10, 1, 13, 3, 10)],
            }
        )
        expected = pd.DataFrame(
            {
                't': [datetime.datetime(2020, 10, 1, 13, 3, 10)],
                'tt': [datetime.datetime(2020, 10, 1, 13, 3, 10)],
            }
        )

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
        self.df = pd.DataFrame(
            {
                'one': np.ones((4,)),
                'zeros': np.zeros((4,)),
                'mixed': np.arange(4),
            }
        )

    def test_dropper(self):
        t = pipeline.ZeroVarianceDropper(verbose=True)
        expected = pd.DataFrame({'mixed': np.arange(4)})

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)

    def test_dropper_fit_higher_variance(self):
        t = pipeline.ZeroVarianceDropper()
        expected = pd.DataFrame({'mixed': np.arange(4)})

        t.fit(self.df)

        df = self.df.copy()
        df.iloc[0, 0] = 0
        with self.assertWarns(Warning):
            result = t.transform(df)

        pd.testing.assert_frame_equal(result, expected)


class TestSignalSorter(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                'one': np.ones((4,)),
                'zeros': np.zeros((4,)),
                'mixed': np.arange(4),
                'binary': [0, 1, 0, 1],
                'cont': [10, 11, 10, 11],
            }
        )

    def test_sorter(self):
        t = pipeline.SignalSorter(verbose=True)
        expected = self.df.loc[:, ['mixed', 'cont', 'one', 'zeros', 'binary']]

        result = t.fit_transform(self.df)

        pd.testing.assert_frame_equal(result, expected)


class TestColumnSorter(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                'one': np.ones((2,)),
                'zeros': np.zeros((2,)),
                'mixed': np.arange(2),
                'binary': [0, 1],
            }
        )

        self.df2 = pd.DataFrame(
            {
                'mixed': np.arange(2),
                'zeros': np.zeros((2,)),
                'one': np.ones((2,)),
                'binary': [0, 1],
            }
        )

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

        self.df_sel = pd.DataFrame(
            {
                'test': [0.0, 1.0, 0.2, 0.3, 0.5],
                'test2': [0.0, 1.0, 0.2, 0.3, 0.5],
            }
        )

    def test_creator(self):
        t = pipeline.DifferentialCreator(columns=['test'])

        expected = pd.DataFrame(
            {
                'test': [0.0, 1.0, 0.2, 0.3, 0.5],
                'test_dif': [0.0, 1.0, -0.8, 0.1, 0.2],
            }
        )

        result = t.fit_transform(self.df)
        pd.testing.assert_frame_equal(result, expected)

    def test_creator_selection(self):
        t = pipeline.DifferentialCreator(columns=['test'])

        expected = pd.DataFrame(
            {
                'test': [0.0, 1.0, 0.2, 0.3, 0.5],
                'test2': [0.0, 1.0, 0.2, 0.3, 0.5],
                'test_dif': [0.0, 1.0, -0.8, 0.1, 0.2],
            }
        )

        result = t.fit_transform(self.df_sel)
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'feature_range,clip,p,exp',
    [
        (
            (0, 1.0),
            None,
            100,
            pd.DataFrame(
                {
                    'a': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                    'b': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                }
            ),
        ),
        (
            (0, 0.5),
            None,
            100,
            pd.DataFrame(
                {
                    'a': [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                    'b': [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                }
            ),
        ),
        (
            (0, 0.5),
            (-1, 1.0),
            100,
            pd.DataFrame(
                {
                    'a': [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0],
                    'b': [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0],
                }
            ),
        ),
        (
            (0, 0.5),
            (0, 1.0),
            100,
            pd.DataFrame(
                {
                    'a': [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                    'b': [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                }
            ),
        ),
    ],
)
def test_ClippingMinMaxScaler(
    feature_range: T.Tuple[float, float],
    clip: T.Optional[T.Tuple[float, float]],
    p: float,
    exp: pd.DataFrame,
):
    scaler = pipeline.ClippingMinMaxScaler(feature_range, clip=clip, p=p)

    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [0, 1, 2, 3]})
    scaler.fit_transform(df)

    df = pd.DataFrame(
        {'a': [-8, -5, -2, 1, 4, 7, 10], 'b': [-9, -6, -3, 0, 3, 6, 9]}
    )
    result = scaler.transform(df)
    pd.testing.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    'features,data,exp',
    [
        (
            [
                {
                    'name': 'area',
                    'features': ['height', 'width'],
                    'op': 'mul',
                },
                {
                    'name': 'AandB',
                    'features': ['a', 'b'],
                    'op': 'and',
                },
                {
                    'name': 'sum',
                    'features': ['height', 'width'],
                    'op': 'add',
                    'keep': False,
                },
                {
                    'name': 'area-sum',
                    'features': ['area', 'sum'],
                    'op': 'sub',
                },
            ],
            pd.DataFrame(
                {
                    'height': [1, 2, 3],
                    'width': [3, 2, 1],
                    'a': [True, False, True],
                    'b': [True, True, False],
                }
            ),
            pd.DataFrame(
                {
                    'height': [1, 2, 3],
                    'width': [3, 2, 1],
                    'a': [True, False, True],
                    'b': [True, True, False],
                    'area': [3, 4, 3],
                    'AandB': [True, False, False],
                    'area-sum': [-1, 0, -1],
                }
            ),
        )
    ],
)
def test_FeatureCreator(
    features: T.List[T.Dict[str, T.Any]],
    data: pd.DataFrame,
    exp: pd.DataFrame,
):
    transformer = pipeline.FeatureCreator(features=features)

    result = transformer.fit_transform(data)

    pd.testing.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    'features,data,exp',
    [
        (
            1,
            pd.DataFrame(),
            f'Expected features to be of type list or set. Got: {type(1)}.',
        ),
        (
            [1],
            pd.DataFrame(),
            'Expected feature at index 0 to be either a dict or '
            f'NewFeatureModel. Got: {type(1)}.',
        ),
        (
            [
                {
                    'name': 'areaX',
                    'features': ['heightX', 'widthX'],
                    'op': 'mul',
                },
            ],
            pd.DataFrame(
                {
                    'height': [1, 2, 3],
                    'width': [3, 2, 1],
                    'a': [True, False, True],
                    'b': [True, True, False],
                }
            ),
            "Missing columns ['heightX', 'widthX'] in input. Available "
            "columns: ['a', 'b', 'height', 'width'].",
        ),
    ],
)
def test_FeatureCreator_fails(
    features: T.List[T.Dict[str, T.Any]],
    data: pd.DataFrame,
    exp: str,
):
    with pytest.raises(ValueError) as exc_info:
        transformer = pipeline.FeatureCreator(features=features)
        transformer.fit_transform(data)

    assert exc_info.value.args[0] == exp
