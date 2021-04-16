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
    """

    df = load_data(path=path, verbose=verbose, **format)
    return temporalize(data=df, timesteps=timesteps, verbose=verbose)


def load_data(*, path: str, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Loads data from `path`.

    Args:
        path (str): Path to csv.
        format (Dict[str, Any]): Keywords for pd.read_csv.

    Returns:
        pd.DataFrame: Returns the loaded data.
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
    """
    data = np.array(data)

    # data is not 2d
    if len(data.shape) > 2:
        return data

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


def detemporalize(*, data: np.array, verbose: bool = True) -> np.array:
    """
    Transforms data from shape rows x timesteps x features to
    (rows * timesteps) x features.

    Arguments:
        data (np.array): Data to transform
        verbose (bool): Whether to print status information.

    Returns:
        np.array: Returns an array of shape (rows * timesteps) x features.

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
