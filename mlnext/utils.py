"""Module with utility functions.
"""
import typing as T

import numpy as np

__all__ = [
    'truncate',
    'check_shape',
    'check_ndim',
    'check_size',
    'rename_keys',
    'RangeDict'
]


def truncate(
    *arrays: T.Tuple[np.ndarray, ...],
    axis: int = 0
) -> T.Iterator[T.Tuple[np.ndarray, ...]]:
    """Truncates the arrays in each tuple to the minimum length of an array in
     the that tuple.

    Args:
        axis (int): Axis to of arrays to truncate.

    Yields:
        T.Iterator[T.Tuple[np.ndarray, ...]]: Iterator over tuple of arrays
         where each tuple has of arrays has the same length.

    Example:
        >>> arr1 = [np.ones((1, 1)), np.zeros((2, 1))]
        >>> arr2 = [np.zeros((2, 1)), np.ones((3, 1))]
        >>> np.hstack(list(mlnext.utils.truncate(*zip(arr1, arr2))))
        ([1, 0, 0], [0, 1, 1])
    """
    for i, array in enumerate(arrays):
        if not isinstance(array, tuple):
            raise ValueError(
                f'Expected tuple or list but got {type(array)} for array at '
                f'position {i}.')

        # find minimum length in tuple
        length = min(map(lambda x: np.shape(x)[axis], array))
        # moveaxis creates a view on the array that moves axis to the front
        yield tuple(map(lambda x: np.moveaxis(x, axis, 0)[:length], array))


def check_shape(
    *arrays: np.ndarray,
    shape: T.Optional[T.Tuple[int, ...]] = None,
    exclude_axis: T.Optional[int] = None
):
    """Checks the shape of one or more arrays. If `shape` is not None, all
    arrays must match `shape`. Otherwise `shape` is set to the shape of the
    first array. With exclude_axis, an axis can be excluded from the check.

    Args:
        shape (T.Optional[T.Tuple[int, ...]], optional): Shape to match. If not
          defined, then the shape of the first array is taken. Defaults to
          None.
        exclude_axis (T.Optional[int], optional): Excludes an axis from the
         check. Shape must be defined without the axis. Defaults to None.

    Raises:
        ValueError: Raised if an array has the wrong shape.

    Example:
        >>> mlnext.utils.check_shape([np.zeros((1, 2)), np.ones((2, 2)),
        ...                           shape=(2,), exclude_axis=0)
    """
    shapes = list(map(lambda x: x.shape, arrays))
    if exclude_axis is not None:
        shapes = list(map(lambda x: tuple(np.delete(x, exclude_axis)), shapes))

    if shape is None:
        shape = shapes[0]

    for i, shape_ in enumerate(shapes):
        if shape_ != shape:
            raise ValueError(
                f'Expected shape {shape} but got shape {shape_} for array at '
                f'position {i} (exclude_axis: {exclude_axis}).')


def check_ndim(
    *arrays: np.ndarray,
    ndim: int,
    strict: bool = True
):
    """Checks whether each passed array has exactly `ndim` number of
    dimensions, if strict is false then the number of dimensions must be at
    most `ndim`.

    Args:
        ndim (int): Number of dimensions.
        strict (bool, optional): If true ndim must match exactly, otherwise
          at most.

    Raises:
        ValueError: Raised if an error does not match the ndim requirements.

    Example:
        >>> mlnext.utils.check_ndim(np.ones((1, 2, 3)), np.ones((3, 2, 1)),
        ...                         ndim=3)
    """

    for i, arr in enumerate(arrays):
        if arr.ndim > ndim or (strict and arr.ndim < ndim):
            raise ValueError(
                f'Expected array of dimension {ndim}, but got {arr.ndim} for '
                f'array at position {i}.')


def check_size(
    *arrays: np.ndarray,
    size: int,
    axis: int,
    strict: bool = True
):
    """Checks whether each array has exactly `size` elements along `axis`,
    if strict is false then it must be at most `size` elements. If strict is
    false and the axis is missing, then the array is ignored.

    Args:
        size (int): Number of elements along the axis.
        axis (int): Axis to check.
        strict (bool, optional): If true the check is exact, otherwise at most.

    Raises:
        ValueError: Raised when the axis is missing or the size requirement
          is not fulfilled.

    Example:
        >>> mlnext.utils.check_size(np.ones((10, 1)), np.ones((10, 3)),
        ...                         size=10, axis=0)
    """

    for i, arr in enumerate(arrays):
        if arr.ndim - 1 < axis:
            if strict:
                raise ValueError(
                    f'Array at position {i} is missing axis {axis}.')
            else:
                continue

        if (shape := arr.shape[axis]) > size or (strict and shape < size):
            raise ValueError(
                f'Expected axis {axis} of array to be of size {size}, but got '
                f'{shape} for array at position {i}.')


def rename_keys(
    mapping: T.Dict[str, T.Any],
    *,
    prefix: T.Optional[str] = None,
    suffix: T.Optional[str] = None
) -> T.Dict[str, T.Any]:
    """Renames every key in `mapping` with a `prefix` and/or `suffix`.

    Args:
        mapping (T.Dict): Mapping.
        prefix (str, optional): String to prepend. Defaults to None.
        suffix (str, optional): String to append. Defaults to None.

    Returns:
        T.Dict: Returns the updated mapping.
    """

    return {
        f'{prefix or ""}{k}{suffix or ""}': v
        for k, v in mapping.items()
    }


def flatten(
    mapping: T.Mapping[str, T.Any],
    *,
    prefix: str = '',
    sep: str = '.',
    flatten_list: bool = True
) -> T.Mapping[str, T.Any]:
    """Turns a nested mapping into a flattened mapping.

    Args:
        mapping (T.Mapping[str, T.Any]): Mapping to flatten.
        prefix (str): Prefix to preprend to the key.
        sep (str): Seperator of flattened keys.
        flatten_list (bool): Whether to flatten lists.

    Returns:
        T.Mapping[str, T.Any]: Returns a (flattened) mapping.

    Example:
        >>> flatten({
        ...    'flat1': 1,
        ...    'dict1': {'c': 1, 'd': 2},
        ...    'nested': {'e': {'c': 1, 'd': 2}, 'd': 2},
        ...    'list1': [1, 2],
        ...    'nested_list': [{'1': 1}]
        ... })
        {
            'flat1': 1,
            'dict1.c': 1,
            'dict1.d': 2,
            'nested.e.c': 1,
            'nested.e.d': 2,
            'nested.d': 2,
            'list1.0': 1,
            'list1.1': 2,
            'nested_list.0.1': 1
        }
    """

    items: T.List[T.Tuple[str, T.Any]] = []
    for k, v in mapping.items():
        key = f'{sep}'.join([prefix, k]) if prefix else k
        if isinstance(v, T.Mapping):
            items.extend(flatten(
                v,
                prefix=key,
                sep=sep,
                flatten_list=flatten_list
            ).items())

        elif isinstance(v, list) and flatten_list:
            for i, v in enumerate(v):
                items.extend(flatten(
                    {str(i): v},
                    prefix=key,
                    sep=sep,
                    flatten_list=flatten_list
                ).items())

        else:
            items.append((key, v))

    return dict(items)


class RangeDict(dict):
    """Dictionary that accepts range keys.
    """

    def __getitem__(self, k__: int) -> T.Any:
        """Gets an item by key `k__` if it is in range of a key.

        Args:
            k__ (int): Key.

        Raises:
            KeyError: Raised if no range match was found.

        Returns:
            T.Any: Returns the item.
        """
        for key in self.keys():
            if k__ in key:
                return super().__getitem__(key)
        raise KeyError(k__)
