""" Module for data preprocessing.
"""
import datetime
import typing as T
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import OneToOneFeatureMixin
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES

__all__ = [
    'ColumnSelector',
    'ColumnDropper',
    'ColumnRename',
    'NaDropper',
    'Clip',
    'DatetimeTransformer',
    'NumericTransformer',
    'TimeframeExtractor',
    'DateExtractor',
    'ValueMapper',
    'Sorter',
    'Fill',
    'TimeOffsetTransformer',
    'ConditionedDropper',
    'ZeroVarianceDropper',
    'SignalSorter',
    'ColumnSorter',
    'DifferentialCreator',
    'ClippingMinMaxScaler'
]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer to select a list of columns by their name.

    Example:
        >>> data = pd.DataFrame({'a': [0], 'b': [0]})
        >>> ColumnSelector(keys=['a']).transform(data)
        pd.DataFrame({'a': [0]})
    """

    def __init__(self, keys: T.List[str]):
        """Creates ColumnSelector.
        Transformer to select a list of columns for further processing.

        Args:
            keys (T.List[str]): T.List of columns to extract.
        """
        self._keys = keys

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extracts the columns from `X`.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns a DataFrame only containing the selected
            features.
        """
        return X.loc[:, self._keys]


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer to drop a list of columns by their name.

    Example:
        >>> data = pd.DataFrame({'a': [0], 'b': [0]})
        >>> ColumnDropper(columns=['b']).transform(data)
        pd.DataFrame({'a': [0]})
    """

    def __init__(
        self,
        *,
        columns: T.Union[T.List[str], T.Set[str]],
        verbose: bool = False
    ):
        """Creates ColumnDropper.
        Transformer to drop a list of columns from the data frame.

        Args:
            keys (list): T.List of columns names to drop.
        """
        self.columns = set(columns)
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Drops a list of columns of `X`.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the dataframe without the dropped features.
        """

        cols = set(X.columns.to_list())
        if len(m := self.columns - cols) > 0:
            warnings.warn(f'Columns {m} not found in dataframe.')

        if self.verbose:
            print(f'New columns: {cols - self.columns}. '
                  f'Removed: {self.columns}.')

        return X.drop(self.columns, axis=1, errors='ignore')


class ColumnRename(BaseEstimator, TransformerMixin):
    """Transformer to rename column with a function.

    Example:
        >>> data = pd.DataFrame({'a.b.c': [0], 'd.e.f': [0]})
        >>> ColumnRename(lambda x: x.split('.')[-1]).transform(data)
        pd.DataFrame({'c': [0], 'f': [0]})
    """

    def __init__(self, mapper: T.Callable[[str], str]):
        """Create ColumnRename.
        Transformer to rename columns by a mapper function.

        Args:
            mapper (lambda): Mapper rename function.

        Example:
            Given column with name: a.b.c
            lambda x: x.split('.')[-1]
            Returns c
        """
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Renames a columns in `X` with a mapper function.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the dataframe with the renamed columns.
        """
        # split the column name
        # use the last item as new name
        return X.rename(columns=self.mapper)


class NaDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops rows with na values.

    Example:
        >>> data = pd.DataFrame({'a': [0, 1], 'b': [0, np.nan]})
        >>> NaDropper().transform(data)
        pd.DataFrame({'a': [0], 'b': [0]})
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna()


class Clip(BaseEstimator, TransformerMixin):
    """Transformer that clips values by a lower and upper bound.

    Example:
        >>> data = pd.DataFrame({'a': [-0.1, 1.2], 'b': [0.5, 0.6]})
        >>> Clip().transform(data)
        pd.DataFrame({'a': [0, 1], 'b': [0.5, 0.6]})
    """

    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        """Creates Clip.
        Transformer that clips a numeric column to the treshold if the
        threshold is exceeded. Works with an upper and lower threshold. Wrapper
        for pd.DataFrame.clip.

        Args:
            lower (float, optional): lower limit. Defaults to 0.
            upper (float, optional): upper limit. Defaults to 1.
        """
        self.upper = upper
        self.lower = lower

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.clip(lower=self.lower, upper=self.upper, axis=0)


class ColumnTSMapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols: T.List[str],
        timedelta: pd.Timedelta = pd.Timedelta(250, 'ms'),
        classes: T.Optional[T.List[str]] = None,
        verbose: bool = False
    ):
        """Creates ColumnTSMapper.
        Expects the timestamp column to be of type pd.Timestamp.

        Args:
            cols (T.List[str]): names of [0] timestamp column,
              [1] sensor names, [2] sensor values.
            timedelta (pd.Timedelta): Timedelta to resample with.
            classes (T.List[str]): T.List of sensor names.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._cols = cols
        self._timedelta = timedelta
        self._verbose = verbose

        if classes is not None:
            self.classes_ = classes

    def fit(self, X, y=None):
        """Gets the unique values in the sensor name column that
        are needed to expand the dataframe.

        Args:
            X (pd.DataFrame): Dataframe.
            y (array-like, optional): Labels. Defaults to None.

        Returns:
            ColumnTSMapper: Returns this.
        """
        classes = X[self._cols[1]].unique()
        self.classes_ = np.hstack(['Timestamp', classes])
        return self

    def transform(self, X):
        """Performs the mapping to equidistant timestamps.

        Args:
            X (pd.DataFrame): Dataframe.

        Raises:
            ValueError: Raised if column is not found in `X`.

        Returns:
            pd.DataFrame: Returns the remapped dataframe.
        """

        # check is fit had been called
        check_is_fitted(self)

        # check if all columns exist
        if not all([item in X.columns for item in self._cols]):
            raise ValueError(
                f'Columns {self._cols} not found in DataFrame '
                f'{X.columns.to_list()}.')

        # split sensors into individual columns
        # create new dataframe with all _categories
        # use timestamp index, to use resample later on
        # initialized with na
        sensors = pd.DataFrame(
            None, columns=self.classes_, index=X[self._cols[0]])

        # group by sensor
        groups = X.groupby([self._cols[1]])

        # write sensor values to sensors which is indexed by the timestamp
        for g in groups:
            sensors.loc[g[1][self._cols[0]], g[0]
                        ] = g[1][self._cols[2]].to_numpy()

        sensors = sensors.apply(pd.to_numeric, errors='ignore')

        # fill na, important before resampling
        # otherwise mean affects more samples than necessary
        # first: forward fill to next valid observation
        # second: backward fill first missing rows
        sensors = sensors.fillna(method='ffill').fillna(method='bfill')

        # resamples to equidistant timeframe
        # take avg if multiple samples in the same timeframe
        sensors = sensors.resample(self._timedelta).mean()
        sensors = sensors.fillna(method='ffill').fillna(method='bfill')

        # FIXME: to avoid nans in model, but needs better fix
        sensors = sensors.fillna(value=0.0)

        # move index to column and use rangeindex
        sensors['Timestamp'] = sensors.index
        sensors.index = pd.RangeIndex(stop=sensors.shape[0])

        if self._verbose:
            start, end = sensors.iloc[0, 0], sensors.iloc[-1, 0]
            print('ColumnTSMapper: ')
            print(f'{sensors.shape[0]} rows. '
                  f'Mapped to {self._timedelta.total_seconds()}s interval '
                  f'from {start} to {end}.')

        return sensors


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Transforms a list of columns to datetime.

    Example:
        >>> data = pd.DataFrame({'dt': ['2021-07-02 16:30:00']})
        >>> data = DatetimeTransformer(columns=['dt']).transform(data)
        >>> data.dtypes
        dt    datetime64[ns]
    """

    def __init__(
        self,
        *,
        columns: T.List[str],
        dt_format: T.Optional[str] = None
    ):
        """Creates DatetimeTransformer.
        Parses a list of column to pd.Timestamp.

        Args:
            columns (list): T.List of columns names.
            dt_format (str): T.Optional format string.
        """
        super().__init__()
        self._columns = columns
        self._format = dt_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Parses `columns` to datetime.

        Args:
            X (pd.DataFrame): Dataframe.

        Raises:
            ValueError: Raised if columns are missing in `X`.

        Returns:
            pd.DataFrame: Returns the dataframe with datetime columns.
        """
        X = X.copy()
        # check if columns in dataframe
        if len(diff := set(self._columns) - set(X.columns)):
            raise ValueError(
                f'Columns {diff} not found in DataFrame with columns'
                f'{X.columns.to_list()}.')

        # parse to pd.Timestamp
        X[self._columns] = X[self._columns].apply(
            lambda x: pd.to_datetime(x, format=self._format), axis=0)
        # column wise

        return X


class NumericTransformer(BaseEstimator, TransformerMixin):
    """Transforms a list of columns to numeric datatype.

    Example:
        >>> data = pd.DataFrame({'a': [0], 'b': ['1']})
        >>> data.dtypes
        a     int64
        b    object
        >>> data = NumericTransformer().transform(data)
        >>> data.dtypes
        a    int64
        b    int64
    """

    def __init__(self, *, columns: T.Optional[T.List[str]] = None):
        """Creates NumericTransformer.
        Parses a list of column to numeric datatype. If None, all are
        attempted to be parsed.

        Args:
            columns (list): T.List of columns names.
            dt_format (str): T.Optional format string.
        """
        super().__init__()
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Parses `columns` to numeric.

        Args:
            X (pd.DataFrame): Dataframe.

        Raises:
            ValueError: Raised if columns are missing in `X`.

        Returns:
            pd.DataFrame: Returns the dataframe with datetime columns.
        """
        X = X.copy()
        # transform all columns
        if self._columns is None:
            columns = X.columns.to_list()
        else:
            columns = self._columns

        if len((diff := list(set(columns) - set(cols := X.columns)))):
            raise ValueError(f'Columns found: {cols.to_list()}. '
                             f'Columns missing: {diff}.')

        # parse to numeric
        # column wise
        X[columns] = X[columns].apply(pd.to_numeric, axis=0)
        return X


class TimeframeExtractor(BaseEstimator, TransformerMixin):
    """Drops sampes that are not between a given start and end time.
    Limits are inclusive.

    Example:
        >>> data = pd.DataFrame(
            {'dates': [datetime.datetime(2021, 7, 2, 9, 50, 0),
                       datetime.datetime(2021, 7, 2, 11, 0, 0),
                       datetime.datetime(2021, 7, 2, 12, 10, 0)],
             'values': [0, 1, 2]})
        >>> TimeframeExtractor(time_column='dates',
                               start_time= datetime.time(10, 0, 0),
                               end_time=datetime.time(12, 0, 0)
                               ).transform(data)
        pd.DataFrame({'dates': datetime.datetime(2021, 7, 2, 11, 0, 0),
                      'values': [1]})
    """

    def __init__(
        self,
        *,
        time_column: str,
        start_time: datetime.time,
        end_time: datetime.time,
        invert: bool = False,
        verbose: bool = False
    ):
        """Creates TimeframeExtractor.
        Drops samples that are not in between `start_time` and  `end_time` in
        `time_column`.

        Args:
            time_column (str): Column name of the timestamp column.
            start_time (datetime.time): Start time.
            end_time (datetime.time): End time.
            invert(bool): Whether to invert the range.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._start = start_time
        self._end = end_time
        self._column = time_column
        self._negate = invert
        self._verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Drops rows from the dataframe if they are not in between
        `start_time` and `end_time`. Limits are inclusive. Reindexes the
        dataframe.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the new dataframe.
        """
        X = X.copy()
        rows_before = X.shape[0]

        dates = pd.to_datetime(X[self._column])
        if self._negate:
            X = X.loc[~((dates.dt.time >= self._start) &
                        (dates.dt.time <= self._end)), :]
        else:
            X = X.loc[(dates.dt.time >= self._start) &
                      (dates.dt.time <= self._end), :]
        X.index = pd.RangeIndex(0, X.shape[0])

        rows_after = X.shape[0]
        if self._verbose:
            print(
                'TimeframeExtractor: \n'
                f'{rows_after} rows. Dropped {rows_before - rows_after} '
                f'rows which are {"in" if self._negate else "not in"} between '
                f'{self._start} and {self._end}.'
            )

        return X


class DateExtractor(BaseEstimator, TransformerMixin):
    """ Drops rows that are not between a start and end date.
    Limits are inclusive.

    Example:
        >>> data = pd.DataFrame(
                {'dates': [datetime.datetime(2021, 7, 1, 9, 50, 0),
                        datetime.datetime(2021, 7, 2, 11, 0, 0),
                        datetime.datetime(2021, 7, 3, 12, 10, 0)],
                'values': [0, 1, 2]})
        >>> DateExtractor(date_column='dates',
                          start_date=datetime.date(2021, 7, 2),
                          end_date=datetime.date(2021, 7, 2)).transform(data)
        pd.DataFrame({'dates': datetime.datetime(2021, 07, 2, 11, 0, 0),
                        'values': [1]})
    """

    def __init__(
        self,
        *,
        date_column: str,
        start_date: datetime.date,
        end_date: datetime.date,
        invert: bool = False,
        verbose: bool = False
    ):
        """Initializes `DateExtractor`.

        Args:
            date_column (str): Name of timestamp column.
            start_date (datetime.date): Start date.
            end_date (datetime.date): End date.
            invert (bool): Whether to invert the range.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._start = start_date
        self._end = end_date
        self._column = date_column
        self._negate = invert
        self._verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Drops rows which date is not between `start` and end date.
        Bounds are inclusive. Dataframe is reindexed.

        Args:
            X (pd.Dataframe): Dataframe.

        Returns:
            pd.Dataframe: Returns the new dataframe.
        """
        rows_before = X.shape[0]

        dates = pd.to_datetime(X[self._column])
        if self._negate:
            X = X.loc[~((dates.dt.date >= self._start) &
                        (dates.dt.date <= self._end)), :]
        else:
            X = X.loc[(dates.dt.date >= self._start) &
                      (dates.dt.date <= self._end), :]
        X.index = pd.RangeIndex(0, X.shape[0])

        rows_after = X.shape[0]
        if self._verbose:
            print(
                'DateExtractor: \n'
                f'{rows_after} rows. Dropped {rows_before - rows_after} rows '
                f'which are {"in" if self._negate else "not in"} between '
                f'{self._start} and {self._end}.'
            )

        return X


class ValueMapper(BaseEstimator, TransformerMixin):
    """Maps values in `column` according to `classes`. Wrapper for
    pd.DataFrame.replace.

    Example:
        >>> data = pd.DataFrame({'a': [0.0, 1.0, 2.0]})
        >>> ValueMapper(columns=['a'], classes={2.0: 1.0}).transform(data)
        pd.DataFrame({'a': [0.0, 1.0, 1.0]})
    """

    def __init__(
        self,
        *,
        columns: T.List[str],
        classes: T.Dict,
        verbose: bool = False
    ):
        """Initialize `ValueMapper`.

        Args:
            columns (T.List[str]): Names of columns to remap.
            classes (T.Dict): Dictionary of old and new value.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._columns = columns
        self._classes = classes
        self._verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Remaps values in `column` according to `classes`.
        Gives UserWarning if unmapped values are found.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the new dataframe with remapped values.
        """
        X = X.copy()
        # warning if unmapped values
        values = pd.unique(X[self._columns].values.ravel('K'))
        if not set(self._classes.keys()).issuperset(values):
            warnings.warn(
                f'Classes {set(self._classes.keys()) - set(values)} ignored.')

        X[self._columns] = X[self._columns].replace(self._classes)
        return X


class Sorter(BaseEstimator, TransformerMixin):
    """Sorts the dataframe by a list of columns. Wrapper for
    pd.DataFrame.sort_values.

    Example:
        >>> data = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
        >>> Sorter(columns=['b'], ascending=True).transform(data)
        pd.DataFrame({'a': [1, 0], 'b': [0, 1]})
    """

    def __init__(
        self,
        *,
        columns: T.List[str],
        ascending: bool = True,
        axis: int = 0
    ):
        """Initialize `Sorter`.

        Args:
            columns (T.List[str]): T.List of column names to sort by.
            ascending (bool): Whether to sort ascending.
            axis (int): Axis to sort by.
        """
        super().__init__()
        self._columns = columns
        self._ascending = ascending
        self._axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Sorts `X` by `columns`.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the sorted Dataframe.
        """
        X = X.copy()
        return X.sort_values(by=self._columns,
                             ascending=self._ascending,
                             axis=self._axis)


class Fill(BaseEstimator, TransformerMixin):
    """Fills NA values with a constant or 'bfill' / 'ffill'.
    Wrapper for df.fillna.

    Example:
        >>> data = pd.DataFrame({'a': [0.0, np.nan]})
        >>> Fill(value=1.0).transform(data)
        pd.DataFrame({'a': [0.0, 1.0]})
    """

    def __init__(
        self,
        *,
        value: T.Any,
        method: T.Optional[str] = None
    ):
        """Initialize `Fill`.

        Args:
            value (T.Any): Constant to fill NAs.
            method (str): method: 'ffill' or 'bfill'.
        """
        super().__init__()
        self._value = value
        self._method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Fills NAs.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the filled dataframe.
        """
        X = X.copy()
        return X.fillna(self._value, method=self._method)


class TimeOffsetTransformer(BaseEstimator, TransformerMixin):
    """`TimeOffsetTransformer` offsets a datetime by `timedelta`.

    Example:
        >>> data = pd.DataFrame(
                {'dates': [datetime.datetime(2021, 7, 1, 16, 0, 0)]})
        >>> TimeOffsetTransformer(time_columns=['dates'],
                                  timedelta=pd.Timedelta(1, 'h')
                                  ).transform(data)
        pd.DataFrame({'dates': datetime.datetime(2021, 07, 2, 17, 0, 0)})
    """

    def __init__(self, *, time_columns: T.List[str], timedelta: pd.Timedelta):
        """
        Initialize `TimeOffsetTransformer`.

        Args:
            time_column (T.List[str]): T.List of names of columns with
              timestamps
            to offset.
            timedelta (pd.Timedelta): Offset.
        """
        super().__init__()
        self._time_columns = time_columns
        self._timedelta = timedelta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Offsets the timestamps in `time_columns` by `timedelta`-

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the dataframe.
        """
        X = X.copy()
        for column in self._time_columns:
            X[column] = pd.to_datetime(X[column]) + self._timedelta
        return X


class ConditionedDropper(BaseEstimator, TransformerMixin):
    """Module to drop rows in `column` that contain numeric values and are
    above `threshold`. If `inverted` is true, values below `threshold` are
    dropped.

    Example:
        >>> data = pd.DataFrame({'a': [0.0, 1.2, 0.5]})
        >>> ConditionedDropper(column='a', threshold=0.5).transform(data)
        pd.DataFrame({'a': [0.0, 0.5]})
    """

    def __init__(
            self,
            *,
            column: str,
            threshold: float,
            invert: bool = False
    ):
        """Initializes `ConditionedDropper`.

        Args:
            column (str): Column to match condition in.
            threshold (float): Threshold.
            inverted (bool, optional): If false, all values below `threshold`
            are dropped, otherwise all values above are dropped.
        """
        super().__init__()
        self.column = column
        self.threshold = threshold
        self.inverted = invert

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Drops rows if below or above a threshold.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the dataframe.
        """
        X = X.copy()

        if not self.inverted:
            X = X.drop(X[X[self.column] > self.threshold].index)
        else:
            X = X.drop(X[X[self.column] < self.threshold].index)

        X.index = pd.RangeIndex(X.shape[0])

        return X


class ZeroVarianceDropper(BaseEstimator, TransformerMixin):
    """Removes all columns that are numeric and have zero variance.
       Needs to be fitted first. Gives a warning if a column that was
       registered as zero variance deviates.

       Example:
            >>> data = pd.DataFrame({'a': [0.0, 0.0], 'b': [1.0, 0.0]})
            >>> ZeroVarianceDropper().fit_transform(data)
            pd.DataFrame({'b': [1.0, 0.0]})
    """

    def __init__(self, verbose: bool = False):
        """Initialize `ZeroVarianceDropper`.

        Args:
            verbose (bool, optional): Whether to print status messages.
        """
        super().__init__()
        self._verbose = verbose

    def _get_zero_variance_columns(self, X: pd.DataFrame) -> T.List[str]:
        """Finds all columns with zero variance.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            T.List[str]: Returns a list of column names.
        """
        var = X.var()
        # get columns with zero variance
        return [k for k, v in var.items() if v == .0]

    def fit(self, X, y=None):
        """Finds all columns with zero variance.

        Args:
            X (pd.DataFrame): Dataframe.
            y (array-like, optional): Labels. Defaults to None.

        Returns:
            ZeroVarianceDropper: Returns self.
        """
        self.columns_ = self._get_zero_variance_columns(X)
        if self._verbose:
            print(
                f'Found {len(self.columns_)} columns with 0 variance '
                f'({self.columns_}).')
        return self

    def transform(self, X):
        """Drops all columns found by fit with zero variance.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the new dataframe.
        """
        check_is_fitted(self, 'columns_')
        X = X.copy()

        # check if columns match
        columns = self._get_zero_variance_columns(X)
        disj = {*columns} ^ {*self.columns_}
        if len(disj) > 0:
            warnings.warn(f'Found column with higher variance: {disj}.')

        before = X.shape[-1]
        X = X.drop(self.columns_, axis=1)
        if self._verbose:
            after = X.shape[-1]
            print(f'Dropped {before - after} columns.')

        return X


class SignalSorter(BaseEstimator, TransformerMixin):
    """Sorts the signals into continuous and binary signals. First the
    continuous, then the binary signals.

    Example:
        >>> data = pd.DataFrame({'a': [0.0, 1.0], 'b': [0.0, 0.2]})
        >>> SignalSorter().fit_transform(data)
        pd.DataFrame({'b': [1.0, 0.0], 'a': [0.0, 1.0]})
    """

    def __init__(self, verbose: bool = False):
        """Initialize `SignalSorter`.

        Args:
            False: [binary, continuous]
            verbose (bool, optional): Whether to print status.
        """
        super().__init__()
        self.verbose = verbose

    def fit(self, X, y=None):
        # find signals that are binary
        uniques = {col: self._is_binary(X[col]) for col in X.columns}
        self.order_ = sorted(uniques.items(), key=lambda v: v[1])

        if self.verbose:
            print(f'Binary: {self.order_}')

        return self

    def _is_binary(self, X: pd.Series) -> bool:
        """
        Args:
            X (pd.Series): Column of a data frame.

        Returns:
            bool: Whether `x` is a binary series.
        """
        unique = X.unique()

        if len(unique) > 2:
            return False

        if len(unique) == 1:
            return True

        try:
            if set(unique.astype('float')) != {1., 0.}:
                return False

            return True
        except Exception:
            return False

    def transform(self, X):
        """Sorts `x` into to a block of continuous and binary signals.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the sorted dataframe.
        """
        check_is_fitted(self, [])
        X = X.copy()
        return X[[c[0] for c in self.order_]]


class ColumnSorter(BaseEstimator, TransformerMixin):
    """Sorts the dataframe in the same order as the fitted dataframe.

    Example:
        >>> data = pd.DataFrame({'a': [0.0, 1.0], 'b': [0.0, 0.2]})
        >>> (sorter := ColumnSorter()).fit(data)
        >>> sorter.transform(pd.DataFrame({'b': [0.2, 1.0], 'a': [0.0, 0.1]}))
        pd.DataFrame({'a': [0.0, 0.1], 'b': [0.2, 1.0]})
    """

    def __init__(self, *, raise_on_error: bool = True, verbose: bool = False):
        """Initialize `ColumnSorter`.

        Attributes:
            raise_on_error (bool): Whether to raise an exception if additional
            columns that were not fitted are found.
            verbose (bool): Whether to print the status.
        """
        super().__init__()
        self.raise_on_error = raise_on_error
        self.verbose = verbose

    def fit(self, X, y=None):
        self.columns_ = X.columns.to_numpy()

        if self.verbose:
            print(f'Sorting in order {self.columns_}.')
        return self

    def transform(self, X):
        """Sorts `X` by `columns`.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the sorted Dataframe.
        """
        check_is_fitted(self)

        if len((diff := list(set(self.columns_) - set(X.columns)))):
            raise ValueError(f'Columns missing: {diff}.')

        if len((diff := list(set(X.columns) - set(self.columns_)))):
            if self.raise_on_error:
                raise ValueError(f'Found additional columns: {diff}.')
            else:
                warnings.warn(f'Found additional columns: {diff}.')

        return X.loc[:, self.columns_]


class DifferentialCreator(BaseEstimator, TransformerMixin):
    """Calculates signal differences between subsequent time points.
    Concatenates the new information with the dataframe.

    Example:
        >>> data = pd.DataFrame({'a': [1.0, 2.0, 1.0]})
        >>> dcreator = DifferentialCreator(columns=['a'])
        >>> dcreator.transform(pd.DataFrame(data)
        pd.DataFrame({'a': [1.0, 2.0, 1.0], 'a_dif': [1.0, -1.0, 0.0]})
    """

    def __init__(self, *, columns: T.List[str]):
        """Initialize `DifferentialCreator`.

        Attributes:
            keys: T.List[str]: Columns to create derivatives
        """
        super().__init__()
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Calculate differences between subsequent points. Fill NaN with zero.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the concatenated DataFrame.
        """
        X_dif = (X[self._columns]
                 .diff(axis=0)
                 .fillna(0)
                 .add_suffix('_dif'))
        return pd.concat([X, X_dif], axis=1)


class ClippingMinMaxScaler(
        OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Normalizes the fitted data to the interval ``feature_range``. The
    parameter ``p`` can be used to calculate the ``max`` value as the ``p``-th
    percentile of the fitted data, i.e., ``p``% of the data is below.
    Data which exceeds the limits of ``feature_range`` after the scaling can be
    clipped to specific values via a ``clip`` range.

    Example:
        >>> data = pd.DataFrame({'a': [1, 2, 3, 4]})
        >>> scaler = mlnext.ClippingMinMaxScaler(
        ...     feature_range=(0, 0.5),
        ...     clip=(0, 1))
        >>> scaler.fit_transform(df)
            a
        0	0.000000
        1	0.166667
        2	0.333333
        3	0.500000
        >>> df2 =  pd.DataFrame({'a': [1, 4, 6, 8, 10]})
                a
        0	0.000000
        1	0.500000
        2	0.833333
        3	1.000000
        4	1.000000
    """

    _parameter_constraints: T.Dict[str, list] = {
        'feature_range': [tuple, list],
        'copy': ['boolean'],
        'clip': [None, tuple, list],
        'p': [int, float]
    }

    def __init__(
        self,
        feature_range: T.Tuple[float, float] = (0, 1),
        *,
        clip: T.Optional[T.Tuple[float, float]] = None,
        p: float = 100.,
        copy: bool = True
    ):
        """Initializes `ClippingMinMaxScaler`.

        Args:
            feature_range (T.Tuple[float, float]): New feature min and max.
              Defaults to (0., 1.).
            clip (T.Tuple[float, float]): Range to clip values. Defaults to
              None.
            p (float): Percentile of data that is used as data maximum.
              Defaults to 100.
            copy (bool, optional): Whether to create a copy. Defaults to True.
        """

        self.feature_range = feature_range
        self.clip = clip
        self.p = p
        self.copy = copy

    def fit(self, X, y=None):
        """Fits the scaler to the data.

        Args:
            X (np.array): Data.
            y ([type], optional): Unused.

        Returns:
            MinMaxScaler: Returns self.
        """
        # FIXME: hack to preserve output format
        # update if output_format can be preserved through sklearn
        if isinstance(X, pd.DataFrame):
            self.set_output(transform='pandas')

        self._validate_params()

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            reset=True
        )

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.percentile(X, self.p, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        f_range = self.feature_range
        self.scale_ = (f_range[1] - f_range[0]) / self.data_range_
        self.min_ = f_range[0] - self.data_min_ * self.scale_

        return self

    def transform(self, X) -> np.ndarray:
        """Transforms ``X`` to the new feature range.

        Args:
            X (np.array): Data.

        Returns:
            np.array: Returns the scaled ``X``.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X,
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            reset=False
        )

        X *= self.scale_
        X += self.min_

        if self.clip is not None:
            X = np.clip(X, self.clip[0], self.clip[1])

        return X
