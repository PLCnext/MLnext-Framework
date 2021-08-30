import os
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd


def load_data_3d(*,
                 path: str,
                 timesteps: int,
                 format: Dict[str, Any] = {},
                 verbose: bool = True) -> np.array:
    """Loads data from `path` and temporalizes it with `timesteps`.

    Args:
        path (str): Path to file.
        timesteps (int): Widow size.
        format (Dict[str, Any]): Format args for pd.read_csv.
        verbose (bool): Whether to print status information.

    Returns:
        np.array: Returns the data.

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
        format (Dict[str, Any]): Keywords for pd.read_csv.

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


def temporalize(*,
                data: Union[pd.DataFrame, np.array],
                timesteps: int,
                verbose: bool = True) -> np.array:
    """ Transforms data from in_rows x features to out_rows x timesteps x
    features. If timesteps is not a proper divisor of rows, the superfluous
    rows are discarded.

    Arguments:
        data (pd.DataFrame, np.array): Data to transform.
        timesteps (int): Number of timesteps.
        verbose (bool): Whether to print status information.

    Returns:
        np.array: Returns an array of shape rows x timesteps x features.

    Example:
        >>> # Transform 2d data into 3d
        >>> data = np.zeros((6, 2))
        >>> temporalize(data=data, timesteps=2)
        Dropped 0 rows. New shape: (3, 2, 2).
    """
    data = np.array(data)

    rows = data.shape[0] // timesteps
    features = data.shape[-1]

    # timesteps has to be a proper divisor of data row count
    # end is the index of the last usable row
    discard = (data.shape[0] % timesteps)
    end = data.shape[0] - discard

    data = np.reshape(data[:end], (rows, timesteps, features))

    if verbose:
        print(f'Dropped {discard} rows. New shape: {data.shape}.')

    return data


def detemporalize(data: np.array, *, verbose: bool = True) -> np.array:
    """
    Transforms data from shape rows x timesteps x features to
    (rows * timesteps) x features.

    Arguments:
        data (np.array): Data to transform
        verbose (bool): Whether to print status information.

    Returns:
        np.array: Returns an array of shape (rows * timesteps) x features.

    Example:
        >>> # Transform 3d data into 2d
        >>> data = np.zeros((3, 2, 2))
        >>> detemporalize(data)
        Old shape: (3, 2, 2). New shape: (6, 2).
    """
    data = np.array(data)
    shape = np.shape(data)

    if len(shape) <= 2:
        return data

    rows = shape[0] * shape[1]
    features = shape[2]

    if verbose:
        print(f'Old shape: {shape}. New shape: ({rows}, {features}).')

    return np.reshape(data, (rows, features))


def sample_normal(*, mean: np.array, std: np.array) -> np.array:
    """Samples from a normal gaussian with mu=`mean` and sigma=`std`.

    Args:
        mean (np.array): Mean of the normal distribution.
        std (np.array): Standard deviation of the normal distribution.

    Returns:
        np.array: Returns the drawn samples.

    Example:
        >>> # Sample from a normal distribution with mean and standard dev.
        >>> sample_normal(mean=[0.1], std=[1])
        array([-0.77506174])
    """
    return np.random.normal(loc=mean, scale=std)


def sample_bernoulli(mean: np.array) -> np.array:
    """Samples from a bernoulli distribution with `mean`.

    Args:
        mean (np.array): Mean of the bernoulli distribution.

    Returns:
        np.array: Returns the drawn samples.

    Example:
        >>> # Sample from a bernoulli distribution with mean
        >>> sample_bernoulli(mean=0.2)
        0
    """
    return np.random.binomial(n=1, p=mean)
