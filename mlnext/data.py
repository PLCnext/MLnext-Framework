""" Module for data loading and manipulation.
"""
import os
import typing as T
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .utils import check_ndim

__all__ = [
    'load_data_3d',
    'load_data',
    'temporalize',
    'detemporalize',
    'sample_normal',
    'sample_bernoulli'
]


def load_data_3d(
    path: str,
    *,
    timesteps: int,
    format: T.Dict[str, T.Any] = {},
    verbose: bool = True
) -> np.ndarray:
    """Loads data from `path` and temporalizes it with `timesteps`.

    Args:
        path (str): Path to file.
        timesteps (int): Widow size.
        format (T.Dict[str, T.Any]): Format args for pd.read_csv.
        verbose (bool): Whether to print status information.

    Returns:
        np.ndarray: Returns the data.

    Example:
        >>> # Loading 2d data and reshaping it to 3d
        >>> X_train = load_data_3d(path='./data/train.csv', timesteps=10)
        >>> X_train.shape
        (100, 10, 18)
    """

    df = load_data(path=path, verbose=verbose, **format)
    return temporalize(data=df, timesteps=timesteps, verbose=verbose)


def load_data(path: str, *, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Loads data from `path`.

    Args:
        path (str): Path to csv.
        format (T.Dict[str, T.Any]): Keywords for pd.read_csv.

    Returns:
        pd.DataFrame: Returns the loaded data.

    Example:
        >>> # Loading data from a csv file with custom seperator
        >>> data = load_data('./data/train.csv', sep=',')
        Loaded train.csv with 1000 rows and 18 columns.
    """
    df = pd.read_csv(path, **kwargs)

    if verbose:
        _, name = os.path.split(path)
        rows, cols = df.shape
        print(f'Loaded {name} with {rows} rows and {cols} columns.')

    return df


def temporalize(
    data: T.Union[pd.DataFrame, np.ndarray],
    *,
    timesteps: int,
    stride: int = 0,
    verbose: bool = False
) -> np.ndarray:
    """Transforms a 2 dimensional array (rows, features) into a 3 dimensional
    array of shape (new_rows, timesteps, features). The step size along axis 0
    can be set with ``stride``. If ``stride=0`` or ``stride=timesteps``, the
    operation is equivalent to ``data.reshape(-1, timesteps, features)``.
    Note: if rows % timesteps  != 0 some rows might be discarded.

    Arguments:
        data (pd.DataFrame, np.ndarray): Data to transform.
        timesteps (int): Number of timesteps.
        stride (int): Step size along the first axis (Default: 0).
        verbose (bool): Whether to print status information.

    Returns:
        np.ndarray: Returns an array of shape rows x timesteps x features.

    Example:
        >>> import numpy as np
        >>> import mlnext

        >>> # setup data
        >>> i, j = np.ogrid[:6, :3]
        >>> data = 10 * i + j
        >>> print(data)
        [[ 0  1  2]
         [10 11 12]
         [20 21 22]
         [30 31 32]
         [40 41 42]
         [50 51 52]]

        >>> # Transform 2d data into 3d
        >>> mlnext.temporalize(data=data, timesteps=2, verbose=True)
        Old shape: (6, 2). New shape: (3, 2, 3).
        [[[ 0  1  2]
          [10 11 12]]
          [[20 21 22]
           [30 31 32]]
          [[40 41 42]
           [50 51 52]]]

        >>> # Transform 2d into 3d with stride=1
        >>> mlnext.temporalize(data, timesteps=3, stride=1, verbose=True)
        Old shape: (6, 3). New shape: (4, 3, 3).
        [[[ 0  1  2]
          [10 11 12]
          [20 21 22]]
         [[10 11 12]
          [20 21 22]
          [30 31 32]]
         [[20 21 22]
          [30 31 32]
          [40 41 42]]
         [[30 31 32]
          [40 41 42]
          [50 51 52]]]

    """
    data = np.array(data)
    old_shape = data.shape

    check_ndim(data, ndim=2)

    if timesteps < 1:
        raise ValueError('Timesteps must be greater than 1.')

    if stride < 0:
        raise ValueError('Stride must be greater than 0.')

    if stride > timesteps:
        warnings.warn(
            f'Reversion with mlnext.detemporalize will result in a loss of '
            f'rows (stride: {stride} larger than timesteps: {timesteps}).')

    # stride = 0 and stride=timesteps is the same as a simple reshape
    # to (rows, timesteps, features) (slice=0 is replaced by timesteps)
    stride = stride or timesteps

    # sliding view with stride
    data = sliding_window_view(
        data,
        window_shape=(timesteps, data.shape[-1]),
    ).squeeze(axis=1)[::stride]

    if verbose:
        print(f'Old shape: {old_shape}. New shape: {data.shape}.')

    return data


def detemporalize(
    data: np.ndarray,
    *,
    stride: int = 0,
    last_point_only: bool = False,
    verbose: bool = False
) -> np.ndarray:
    """
    Transforms a 3 dimensional array (rows, timesteps, features) into a 2
    dimensional array (new_rows, features). If ``stride`` >= timesteps
    or 0, then the operation is equivalent to ``data.reshape(-1, features)``
    and new_rows equals rows * timesteps. If 0 < ``stride`` < timesteps, the
    stride induced elements will be removed and new_rows equals (rows -
    timesteps) * timesteps. If ``last_point_only=True`` then only the last
    point in each window is kept and new_rows equals (rows, features).

    Arguments:
        data (np.ndarray): Array to transform.
        stride (np.ndarray): Stride that was used to transform the array from
          2d into 3d.
        last_point_only (np.ndarray): Whether to only take the last point of
          each window.
        verbose (bool): Whether to print old and new shape.

    Returns:
        np.ndarray: Returns an array of shape (rows * timesteps) x features.

    Example:
        >>> import numpy as np
        >>> import mlnext

        >>> # setup data
        >>> i, j = np.ogrid[:6, :3]
        >>> data = 10 * i + j
        >>> print(data)
        [[ 0  1  2]
         [10 11 12]
         [20 21 22]
         [30 31 32]
         [40 41 42]
         [50 51 52]]

        >>> # Transform 3d data into 2d
        >>> data_3d = mlnext.temporalize(data, timesteps=2)
        >>> print(data_3d)
        [[[ 0  1  2]
          [10 11 12]]
         [[20 21 22]
           [30 31 32]]
         [[40 41 42]
          [50 51 52]]]
        >>> mlnext.detemporalize(data_3d, verbose=True)
        Old shape: (3, 2, 3). New shape: (6, 3).
        [[ 0  1  2]
         [10 11 12]
         [20 21 22]
         [30 31 32]
         [40 41 42]
         [50 51 52]]

        >>> # Transform 3d data into 2d with stride=1
        >>> data_3d = mlnext.temporalize(data,
        ... timesteps=3, stride=1, verbose=True)
        Old shape: (6, 3). New shape: (4, 3, 3).
        >>> print(data_3d)
        [[[ 0  1  2]
          [10 11 12]
          [20 21 22]]
         [[10 11 12]
          [20 21 22]
          [30 31 32]]
         [[20 21 22]
          [30 31 32]
          [40 41 42]]
         [[30 31 32]
          [40 41 42]
          [50 51 52]]]
        >>> mlnext.detemporalize(data_3d, stride=1, verbose=True)
        Old shape: (4, 3, 3). New shape: (6, 3).
        [[ 0  1  2]
         [10 11 12]
         [20 21 22]
         [30 31 32]
         [40 41 42]
         [50 51 52]]
        >>> # Take only the last point from each window
        >>> mlnext.detemporalize(data_3d, last_point_only=True, verbose=True)
        Old shape: (4, 3, 3). New shape: (4, 3).
        [[20 21 22]
         [30 31 32]
         [40 41 42]
         [50 51 52]]

    """
    data = np.array(data)

    if data.ndim < 3:
        # nothing to do
        return data

    check_ndim(data, ndim=3)
    rows, timesteps, features = data.shape  # (rows, timesteps, features)

    if stride < 0:
        raise ValueError('Stride must be greater than 0.')

    if last_point_only:
        # take only the last point in each window
        s = slice(timesteps - 1, None, timesteps)
        data = data.reshape(-1, features)[s]
    else:
        # remove stride
        step = stride if stride > 0 and stride < timesteps else timesteps
        # extract the last window, we need all of it
        lw = data[-1]
        # take the first `step`-values of each window
        data = data[:-1, :step, :].reshape(-1, features)
        # concat along axis 0
        data = np.r_[data, lw]

    if verbose:
        print(f'Old shape: {(rows, timesteps, features)}. '
              f'New shape: {data.shape}.')

    return data


def sample_normal(*, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Samples from a normal gaussian with mu=`mean` and sigma=`std`.

    Args:
        mean (np.ndarray): Mean of the normal distribution.
        std (np.ndarray): Standard deviation of the normal distribution.

    Returns:
        np.ndarray: Returns the drawn samples.

    Example:
        >>> # Sample from a normal distribution with mean and standard dev.
        >>> sample_normal(mean=[0.1], std=[1])
        array([-0.77506174])
    """
    return np.random.normal(loc=mean, scale=std)


def sample_bernoulli(mean: np.ndarray) -> np.ndarray:
    """Samples from a bernoulli distribution with `mean`.

    Args:
        mean (np.ndarray): Mean of the bernoulli distribution.

    Returns:
        np.ndarray: Returns the drawn samples.

    Example:
        >>> # Sample from a bernoulli distribution with mean
        >>> sample_bernoulli(mean=0.2)
        0
    """
    return np.random.binomial(n=1, p=mean)
