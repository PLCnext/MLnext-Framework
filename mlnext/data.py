""" Module for data loading and manipulation.
"""
import os
import random
import typing as T
import warnings
from math import floor
from random import randrange

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split

from .utils import check_ndim

__all__ = [
    'load_data_3d',
    'load_data',
    'temporalize',
    'detemporalize',
    'sample_normal',
    'sample_bernoulli',
    'train_val_test_split'
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


def train_val_test_split(
    data: pd.DataFrame,
    test_size: float = 0.25,
    val_size: float = 0.25,
    random_state: int = 0,
    shuffle: bool = False,
    anomaly_density: float = 0.5,
    anomaly_proba: float = 0.5,
    anomaly_length_min: int = 1,
    anomaly_length_max: int = 2,
    variance: float = 0.1
) -> T.Dict[str, pd.DataFrame]:
    """Splits data into three data sets for training, validation and test
    while adding anomalies in the test set. For continous features, Gaussian
    noise with mean 0 and parametrized varance is added. For boolean features,
    an anomaly represents a value flipping.

     Args:
        data (pd.DataFrame): Data to split and manipulate.
        test_size (float): Should be between 0.0 and 1.0 and represent the
          proportion of the dataset to include in the test set.
        val_size (float): Should be between 0.0 and 1.0 and represent the
          proportion of the dataset to include in the validation set.
        random_state (int): Controls the shuffling applied to the data before
          applying the split. Pass an int for reproducible output across
          multiple function calls.
        shuffle (bool): Whether or not to shuffle the data before splitting.
        anomaly_density (float): Should be between 0.0 and 1.0 and represent
          the number of chunks for anomalies in the test set. At 0.0, only one
          anomaly chunk is created. At 1.0, the maximum amound of possible
          anomaly chunks is created.
        anomaly_proba (float): Probaility of a chunk containing an anomaly.
        anomaly_length_min (int): Minimum size of an anomaly.
        anomaly_length_max (int): Maximum size of an anomaly.
        variance (float): Variance used for the Gaussian noise.

    Returns:
         T.Dict[str, pd.DataFrame]:  Returns a dict with data sets and labels.

    Example:
        >>> # Create data sets for training, validation and testing
        >>> data = pd.DataFrame(np.arange(0,9), columns=['data'])
        >>> print(data)
           data
        0     0
        1     1
        2     2
        3     3
        4     4
        5     5
        6     6
        7     7
        8     8
        >>> result = train_val_test_split(data)
        >>> print(result)
        {'X_train':data
        0     0
        1     1
        2     2
        3     3,

        'X_val':data
        4     4
        5     5,

        'X_test':data
        6  6.477456
        7  7.000000
        8  8.000000,

        'y_train':Label
        0    0.0
        1    0.0
        2    0.0
        3    0.0,

        'y_val':Label
        0    0.0
        1    0.0,

        'y_test':Label
        0    1.0
        1    0.0
        2    0.0}
    """

    # disable chained assignment warning since behaviour is intended
    pd.options.mode.chained_assignment = None
    # split the data into train, validation, and test sets
    X_train, X_test = train_test_split(data,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=shuffle)
    X_train, X_val = train_test_split(X_train,
                                      test_size=val_size,
                                      random_state=random_state,
                                      shuffle=shuffle)
    # create labels
    y_train = pd.DataFrame(np.zeros(len(X_train)), columns=['Label'])
    y_val = pd.DataFrame(np.zeros(len(X_val)), columns=['Label'])
    y_test = pd.DataFrame(np.zeros(len(X_test)), columns=['Label'])
    # calculate number of possible chunks
    num_possible_chunks = floor(X_test.shape[0] / anomaly_length_max)
    # calculate number of chunks in which the data is distributed
    num_chunks = int(max(1, num_possible_chunks * anomaly_density))
    # calculate the size per chunk
    chunk_size = int(X_test.shape[0]/num_chunks)

    for i in range(num_chunks):
        # draw if anomaly exists in chunk
        if random.random() < anomaly_proba:
            # generate random anomaly size
            anomaly_size = randrange(anomaly_length_min, anomaly_length_max)
            #  define start and end of chunk piece
            start = chunk_size * i
            end = start + chunk_size
            #  manipulate the data
            anomaly = add_noise(X_test.iloc[start:end, :],
                                y_test.iloc[start:end, :],
                                anomaly_size,
                                variance)
            # insert anomalous data
            X_test.iloc[start:end, :] = anomaly['data']
            y_test.iloc[start:end, :] = anomaly['label']

    data = {'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
            }

    return data


def add_noise(
        data: pd.DataFrame,
        label: pd.DataFrame,
        length: int = 1,
        variance: float = 1.0
) -> T.Dict[str, pd.DataFrame]:
    """Splits data into three data sets for training, validation and test
    while adding anomalies in the test set. For continous features, Gaussian
    noise with mean 0 and parametrized varance is added. For boolean features,
    an anomaly represents a value flipping.

     Args:
        data (pd.DataFrame): Data to manipulate.
        data (pd.DataFrame): Label to indicate anomaly position.
        length (int): Length of the anomly.
        variance (float): Variance used for the Gaussian noise.

    Returns:
         T.Dict[str, pd.DataFrame]: Returns the manipulated data and labels.

    Example:
        >>> # Manipluate the data and set the corresponding labels
        >>> data = pd.DataFrame(np.arange(0,5), columns=['data'])
        >>> print(data)
           data
        0     0
        1     1
        2     2
        3     3
        4     4
        >>> label = pd.DataFrame(np.zeros(5), columns=['Label'])
        >>> print(label)
            Label
        0    0.0
        1    0.0
        2    0.0
        3    0.0
        4    0.0
        >>> result = add_noise(data, label, 3)
        >>> print(result)
        {'data':        data
        0  0.940744
        1  1.214670
        2  3.406815
        3  3.000000
        4  4.000000,
        'label':    Label
        0    1.0
        1    1.0
        2    1.0
        3    0.0
        4    0.0}
    """

    for col in data:
        # create random noise for continous features
        if len(np.unique(data[col])) > 2:
            size = data[col][0:length].size
            noise = np.random.normal(0, variance, size=size)
            data[col][0:length] = data[col][0:length] + noise
            label['Label'][0:length] = 1
        # switch labels for bollean features
        else:
            data[col][0:length] = data[col][0:length] * - \
                1 + sum(np.unique(data[col]))
            label['Label'][0:length] = 1

    return {'data': data, 'label': label}
