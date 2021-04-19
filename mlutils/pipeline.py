import datetime
import warnings
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys: List[str]):
        """Creates ColumnSelector.
        Transformer to select a list of columns for further processing.

        Args:
            keys (List[str]): List of columns to extract.
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
    def __init__(self,
                 *,
                 columns: Union[List[str], Set[str]],
                 verbose: bool = False):
        """Creates ColumnDropper.
        Transformer to drop a list of columns from the data frame.

        Args:
            keys (list): List of columns names to drop.
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
    def __init__(self, mapper: Callable[[str], str]):
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
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna()


class Clip(BaseEstimator, TransformerMixin):
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
    def __init__(self,
                 cols: List[str],
                 timedelta: pd.Timedelta = pd.Timedelta(250, 'ms'),
                 classes: List[str] = None,
                 verbose: bool = False):
        """Creates ColumnTSMapper.
        Expects the timestamp column to be of type pd.Timestamp.

        Args:
            cols (List[str]): names of [0] timestamp column, [1] sensor names,
            [2] sensor values.
            timedelta (pd.Timedelta): Timedelta to resample with.
            classes (List[str]): List of sensor names.
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
    def __init__(self, *, columns: List[str], dt_format: str = None):
        """Creates DatetimeTransformer.
        Parses a list of column to pd.Timestamp.

        Args:
            columns (list): List of columns names.
            dt_format (str): Optional format string.
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
        # check if columns in dataframe
        if not all([item in X.columns for item in self._columns]):
            raise ValueError(
                f'Columns {self._columns} not found in DataFrame '
                f'{X.columns.to_list()}.')

        # parse to pd.Timestamp
        X.loc[:, self._columns] = X.loc[:, self._columns].apply(
            lambda x: pd.to_datetime(x, format=self._format), axis=0)
        # column wise

        return X


class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, columns: Optional[List[str]] = None):
        """Creates NumericTransformer.
        Parses a list of column to numeric datatype. If None, all are
        attempted to be parsed.

        Args:
            columns (list): List of columns names.
            dt_format (str): Optional format string.
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
        # transform all columns
        if self._columns is None:
            columns = X.columns
        else:
            columns = self._columns

        if len((diff := list(set(columns) - set(X.columns)))):
            raise ValueError(f'Columns: {columns}. '
                             f'Columns missing: {diff}.')

        if len((diff := list(set(X.columns) - set(columns)))):
            raise ValueError(f'Columns: {self._columns}. '
                             f'Found additional columns: {diff}.')

        # parse to numeric
        # column wise
        X.loc[:, columns] = X.loc[:, columns].apply(pd.to_numeric, axis=0)
        return X


class TimeframeExtractor(BaseEstimator, TransformerMixin):
    """Drops sampes that are not between a given start and end time.
    """

    def __init__(self,
                 time_column: str,
                 start_time: datetime.time,
                 end_time: datetime.time,
                 negate: bool = False,
                 verbose: bool = False):
        """Creates TimeframeExtractor.
        Drops samples that are not in between `start_time` and  `end_time` in
        `time_column`.

        Args:
            time_column (str): Column name of the timestamp column.
            start_time (datetime.time): Start time.
            end_time (datetime.time): End time.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._start = start_time
        self._end = end_time
        self._column = time_column
        self._negate = negate
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
    """

    def __init__(self,
                 time_column: str,
                 start: datetime.date,
                 end: datetime.date,
                 negate: bool = False,
                 verbose: bool = False):
        """Initializes `DateExtractor`.

        Args:
            time_column (str): Name of timestamp column.
            start (datetime.date): Start date.
            end (datetime.date): End date.
            negate (bool): Negate condition.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._start = start
        self._end = end
        self._column = time_column
        self._negate = negate
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
    """Maps values in `column` according to `classes`.
    """

    def __init__(self,
                 column: List[str],
                 classes: Dict,
                 verbose: bool = False):
        """Initialize `ValueMapper`.

        Args:
            column (str): Name of columns to remap.
            classes (Dict): Dictionary of old and new value.
            verbose (bool, optional): Whether to allow prints.
        """
        super().__init__()
        self._columns = column
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
                f'{set(values).symmetric_difference(self._classes.keys())} '
                'ignored.')

        X[self._columns] = X[self._columns].replace(self._classes)
        return X


class Sorter(BaseEstimator, TransformerMixin):
    """Sorts the dataframe by a List of columns.
    """

    def __init__(self,
                 columns: List[str],
                 ascending: bool = True,
                 axis: int = 0):
        """Initialize `Sorter`.

        Args:
            columns (List[str]): List of column names to sort by.
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
        return X.sort_values(by=self._columns,
                             ascending=self._ascending,
                             axis=self._axis)


class Resampler(BaseEstimator, TransformerMixin):
    """Resamples a dataframe to to `timedelta` interval with optionally
    extended range defined by `timeframe`.
    """

    def __init__(self,
                 time_column: str,
                 timedelta: pd.Timedelta,
                 timeframe: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                 verbose: bool = False):
        """Initialize `Resampler`.

        Args:
            time_column (str): Name of column with timestamps.
            timedelta (pd.Timedelta): Resample interval.
            timeframe (Optional[Tuple[pd.Timestamp, pd.Timestamp]], optional):
            List of timestamps defining the new start and end times.
            verbose (bool, optional): Whether to print status.
        """
        super().__init__()
        self._column = time_column
        self._timedelta = timedelta
        self._timeframe = timeframe
        self._verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Resamples `X` with `timdelta` as interval. Optionally `timeframe`
        can be used to  extend the to a new start and end date.

        Args:
            X (pd.DataFrame): Dataframe.

        Raises:
            ValueError: Raised if `timeframe` lies in between the dates present
            in `X`.

        Returns:
            pd.DataFrame: Returns the resampled dataframe.
        """

        # use original timeseries as index
        index = pd.to_datetime(X[self._column])
        data = X.copy()
        data[self._column] = 0.0

        # use a new dataframe for the correct timeframe
        samples = data.set_index(index)
        # make sure everything is numeric, otherwise resample takes ages
        samples = samples.apply(pd.to_numeric, axis=1)

        if self._timeframe is not None:
            # check range
            if ((index.iloc[0] < self._timeframe[0]) |
                    (index.iloc[-1] > self._timeframe[1])):
                raise ValueError('Invalid range.')

            # insert new start and end
            for time in self._timeframe:
                samples.loc[time] = 0.0

        # resample with resample_interval
        samples = samples.resample(self._timedelta).max()

        # move index to column and use rangeindex
        samples[self._column] = samples.index
        samples.index = pd.RangeIndex(stop=samples.shape[0])

        return samples


class TimeValueExpander(BaseEstimator, TransformerMixin):
    """Expands `trigger` in `value_columns` by `timedelta`.
    Dates are given in `time_column`. Mode is either 'f' for forwards or
    'b' for backwards.
    """

    def __init__(self,
                 time_column: str,
                 value_columns: List[str],
                 timedelta: pd.Timedelta,
                 trigger: Any,
                 mode: str = 'b',
                 verbose: bool = False):
        """Initialize `TimeValueExpander`.

        Args:
            time_column (str): Name of column with timestamp.
            value_columns (List[str]): List of columns to expand,
            timedelta (pd.Timedelta): Timedelta to expand to.
            trigger (Any): Trigger value.
            mode (str, optional): Mode. Either 'b' or 'f'.
            verbose (bool, optional): Whether to print status.

        Raises:
            ValueError: Raised if illegal mode is passed.
        """
        super().__init__()
        self._time_column = time_column
        self._value_columns = value_columns
        self._timedelta = timedelta
        self._trigger = trigger
        self._verbose = verbose

        allowed_modes = ['b', 'f']
        if mode not in allowed_modes:
            raise ValueError(
                f'Invalid mode: {mode}. Allowed modes: {allowed_modes}.')

        self._mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Expand `trigger` in `value_column` by time given by `timedelta` and
        `time_column`.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the dataframe.
        """
        X[self._time_column] = pd.to_datetime(X[self._time_column])
        for col in self._value_columns:
            # find positions of 1s
            pos = X[X[col] == self._trigger]

            # get time
            times = pos[self._time_column]

            #
            if self._mode == 'b':
                off_times = times - self._timedelta
                zipped = zip(off_times, times)
            else:
                off_times = times + self._timedelta
                zipped = zip(times, off_times)

            for s, e in zipped:
                X.loc[(X[self._time_column] >= s) &
                      (X[self._time_column] <= e), col] = self._trigger

        return X


class Fill(BaseEstimator, TransformerMixin):
    """Fills NA values with a constant or 'bfill' / 'ffill'.
    Wrapper for df.fillna.
    """

    def __init__(self,
                 value: Optional[Any],
                 columns: Optional[List[str]] = None,
                 method: str = None):
        """Initialize `Fill`.

        Args:
            value (Optional[Any]): Constant to fill NAs.
            columns (Optional[List[str]], optional): Columns to fill. If None
            all columns are filled.
            method (str, optional): Optional method: 'ffill' or 'bfill'.
        """
        super().__init__()
        self._value = value
        self._method = method
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Fills NAs.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            pd.DataFrame: Returns the filled dataframe.
        """
        return X.fillna(self._value, method=self._method)


class TimeOffsetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_columns: List[str], timedelta: pd.Timedelta):
        """
        Initialize `TimeOffsetTransformer`.

        Args:
            time_column (List[str]): List of names of columns with timestamps
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
        for column in self._time_columns:
            X[column] = pd.to_datetime(X[column]) + self._timedelta
        return X


# TODO: finish or remove
class LabelFilter(BaseEstimator, TransformerMixin):
    def __init__(self, time_column: str, timedelta: pd.Timedelta):
        super().__init__()
        self._time_column = time_column
        self._timedelta = timedelta

    def fit(self, X, y=None):
        if y is None:
            raise ValueError('Expected y.')

        # find timedeltas in y, sparse labels
        # expects the timestamps in the first column
        y = pd.to_datetime(y.iloc[:, 0])
        self._deltas = pd.DataFrame(y[1:].to_numpy() - y[:-1].to_numpy())

        # check where deltas exceed timedelta
        exceeded = self._deltas > self._timedelta

        # index  and index + 1 to get timeframes, where timedelta is exceeded

        print(self._deltas)
        print(exceeded)

        return self

    def transform(self, X):
        pass


class ConditionedDropper(BaseEstimator, TransformerMixin):
    """Module to drop rows in `column` that contain numeric values and are above
    `threshold`. If `inverted` is true, values below `threshold` are dropped.
    """

    def __init__(self, column: str, threshold: float, inverted: bool = False):
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
        self.inverted = inverted

    def fit(self, X, y=None):
        return self

    def transform(self, X):

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
    """

    def __init__(self, verbose: bool = False):
        """Initialize `ZeroVarianceDropper`.

        Args:
            verbose (bool, optional): Whether to print status messages.
        """
        super().__init__()
        self._verbose = verbose

    def _get_zero_variance_columns(self, X: pd.DataFrame) -> List[str]:
        """Finds all columns with zero variance.

        Args:
            X (pd.DataFrame): Dataframe.

        Returns:
            List[str]: Returns a list of column names.
        """
        var = X.var()
        # get columns with zero variance
        return [k for k, v in var.iteritems() if v == .0]

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
    """Sorts the signals into continuous and binary signals.
    """

    def __init__(self, continuous_first: bool = True, verbose: bool = False):
        """Initialize `SignalSorter`.

        Args:
            continuous_first (bool, optional): True: [continuous, binary],
            False: [binary, continuous]
            verbose (bool, optional): Whether to print status.
        """
        super().__init__()
        self.continuous_first = continuous_first
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

        # FIXME: might be dangerous
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
        return X[[c[0] for c in self.order_]]


class ColumnSorter(BaseEstimator, TransformerMixin):
    """Sorts the dataframe in the same order as the fitted dataframe.
    """

    def __init__(self, *, raise_on_error: bool = True, verbose: bool = False):
        """Initialize `ColumnSorter`.

        Attributes:
            raise_on_error (bool): Whether to raise an exception if
            differences between the fitted and transforming dataset.
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
            if self.raise_on_error:
                raise ValueError(f'Columns missing: {diff}.')
            else:
                warnings.warn(f'Columns missing: {diff}.')

        if len((diff := list(set(X.columns) - set(self.columns_)))):
            if self.raise_on_error:
                raise ValueError(f'Found additional columns: {diff}.')
            else:
                warnings.warn(f'Found additional columns: {diff}.')

        return X.loc[:, self.columns_]
