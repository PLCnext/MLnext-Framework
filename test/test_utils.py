import typing as T

import numpy as np
import pytest

from mlnext.utils import check_ndim
from mlnext.utils import check_shape
from mlnext.utils import check_size
from mlnext.utils import truncate


@pytest.mark.parametrize(
    'arrays,axis,exp',
    [
        ([(np.ones(1), np.ones(2)), (np.ones(4), np.ones(3))], 0,
         [(np.ones(1), np.ones(1)), (np.ones(3), np.ones(3))]),
        ([(np.ones(1), np.ones(2))], 0,
         [(np.ones(1), np.ones(1))]),
        ([(np.ones((1, 1)), np.ones((2, 1))),
          (np.ones((4, 2)), np.ones((5, 2)))], 0,
         [(np.ones((1, 1)), np.ones((1, 1))),
          (np.ones((4, 2)), np.ones((4, 2)))]),
        ([(np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 3)))], 1,
         [(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))])
    ]
)
def test_truncate(
    arrays: T.List[T.Tuple[np.ndarray, ...]],
    axis: int,
    exp: T.List[T.Tuple[np.ndarray, ...]]
):
    result = list(truncate(*arrays, axis=axis))

    np.testing.assert_equal(result, exp)


@pytest.mark.parametrize(
    'arrays,err_msg',
    [
        ([np.zeros(1)],
         'Expected tuple or list but got <class \'numpy.ndarray\'> for array '
         'at position 0.')
    ]
)
def test_truncate_fails(arrays: T.List[T.Any], err_msg: str):

    with pytest.raises(ValueError) as exc_info:
        list(truncate(*arrays))

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'arrays,ndim,position',
    [
        ([[np.ones((2, 2))], 1, 0]),
        ([[np.ones((2, 2))], 3, 0]),
        ([[np.ones((2, 2)), np.ones((1,))], 2, 1]),
    ]
)
def test_check_ndim_strict_fails(
    arrays: T.List[np.ndarray],
    ndim: int,
    position: int
):
    with pytest.raises(ValueError) as exc_info:
        check_ndim(*arrays, ndim=ndim)

    rec_dim = arrays[position].ndim
    err_msg = (f'Expected array of dimension {ndim}, but got {rec_dim} for '
               f'array at position {position}.')

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'arrays,ndim,strict',
    [
        ([[np.ones((2, 2))], 2, True]),
        ([[np.ones((2, 2)), np.ones((1, 4))], 2, True]),
        ([[np.ones((3, 1, 2)), np.ones((1, 4))], 3, False]),
    ]
)
def test_check_ndim(
    arrays: T.List[np.ndarray],
    ndim: int,
    strict: bool
):

    check_ndim(*arrays, ndim=ndim, strict=strict)


@pytest.mark.parametrize(
    'arrays,size,axis,position',
    [
        [[np.ones((4, 1))], 5, 0, 0],
        [[np.ones((4, 1))], 3, 0, 0],
        [[np.ones((4, 1))], 2, 1, 0],
        [[np.ones((4, 2)), np.zeros((2, 1))], 2, 1, 1],
    ]
)
def test_check_size_fails(
    arrays: T.List[np.ndarray],
    size: int,
    axis: int,
    position: int
):

    with pytest.raises(ValueError) as exc_info:
        check_size(*arrays, size=size, axis=axis)

    rec_shape = arrays[position].shape[axis]
    err_msg = (f'Expected axis {axis} of array to be of size {size}, but got '
               f'{rec_shape} for array at position {position}.')

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'arrays,axis,position',
    [
        ([np.ones((2,))], 1, 0),
        ([np.ones((3, 1)), np.ones((2,))], 1, 1)
    ]
)
def test_check_size_fails_axis_missing(
    arrays: T.List[np.ndarray],
    axis: int,
    position: int
):

    with pytest.raises(ValueError) as exc_info:
        check_size(*arrays, size=1, axis=axis)

    err_msg = f'Array at position {position} is missing axis {axis}.'

    assert exc_info.value.args[0] == err_msg


@pytest.mark.parametrize(
    'arrays,size,axis,strict',
    [
        [[np.ones((4, 1))], 5, 0, False],
        [[np.ones((4, 1))], 4, 0, True],
        [[np.ones((4,))], 2, 1, False],
        [[np.ones((4, 2)), np.zeros((2, 1))], 2, 1, False],
    ]
)
def test_check_size(
    arrays: T.List[np.ndarray],
    size: int,
    axis: int,
    strict: bool
):

    check_size(*arrays, size=size, axis=axis, strict=strict)


@pytest.mark.parametrize(
    'arrays,shape,exclude_axis',
    [
        ([np.zeros((1, 2))], (2,), 0),

        ([np.zeros((1, 2)), np.ones((2, 2))], None, 0),
        ([np.zeros((1, 2)), np.ones((2, 2))], (2,), 0),
        ([np.zeros((1, 2, 3)), np.ones((2, 2, 3))], (2, 3), 0),
        ([np.zeros((1, 2, 3)), np.ones((2, 2, 3))], None, 0),

        ([np.zeros((2, 3)), np.ones((2, 2))], None, 1),
        ([np.zeros((1, 2, 3)), np.ones((1, 2, 4))], (1, 2), 2),

        ([np.zeros((2,)), np.ones((2,))], (2,), None),
        ([np.zeros((2, 2)), np.ones((2, 2))], (2, 2), None),
        ([np.zeros((2, 2, 1)), np.ones((2, 2, 1))], (2, 2, 1), None),
    ]
)
def test_check_shape(
    arrays: T.List[np.ndarray],
    shape: T.Optional[T.Tuple[int, ...]],
    exclude_axis: T.Optional[int]
):

    check_shape(*arrays, shape=shape, exclude_axis=exclude_axis)


@pytest.mark.parametrize(
    'arrays,shape,exclude_axis,exp_shape,rec_shape,position',
    [
        ([np.ones((2, 1)), np.ones((1, 2))], None, 1, (2,), (1,), 1),
        ([np.ones((2, 1)), np.ones((1, 2))], None, 0, (1,), (2,), 1),

        ([np.ones((2, 1)), np.ones((1, 2))], (2,), 1, (2,), (1,), 1),
        ([np.ones((2, 1)), np.ones((2, 2))], (2, 2), None, (2, 2), (2, 1), 0),
    ]
)
def test_check_shape_fails(
    arrays: T.List[np.ndarray],
    shape: T.Optional[T.Tuple[int, ...]],
    exclude_axis: T.Optional[int],
    exp_shape: T.Tuple[int, ...],
    rec_shape: T.Tuple[int, ...],
    position: int
):

    with pytest.raises(ValueError) as exc_info:
        check_shape(*arrays, shape=shape, exclude_axis=exclude_axis)

    err_msg = (f'Expected shape {exp_shape} but got shape {rec_shape} for '
               f'array at position {position} (exclude_axis: {exclude_axis}).')

    assert exc_info.value.args[0] == err_msg
