import typing as T

import pandas as pd
import pytest

from mlnext import model


@pytest.mark.parametrize(
    'a,b',
    [
        (
            pd.Series([3, 2, 0, True, True, False, False]),
            pd.Series([1, -1, 1, True, False, True, False]),
        ),
    ],
)
@pytest.mark.parametrize(
    'op,exp',
    [
        (
            model.LogicalOperation.OR,
            pd.Series([True, True, True, True, True, True, False]),
        ),
        (
            model.LogicalOperation.AND,
            pd.Series([True, True, False, True, False, False, False]),
        ),
        (
            model.LogicalOperation.XOR,
            pd.Series([True, True, True, False, True, True, False]),
        ),
    ],
)
def test_LogicalOperator(
    a: pd.Series,
    b: pd.Series,
    op: model.LogicalOperation,
    exp: pd.Series,
):
    result = op(a, b)

    pd.testing.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    'a,b',
    [
        (
            pd.Series([3, 2, 0, 2.4, 2, 1]),
            pd.Series([1, -1, 1, 0.1, 5, 1]),
        ),
    ],
)
@pytest.mark.parametrize(
    'op,exp',
    [
        (
            model.NumericalOperation.ADD,
            pd.Series([4, 1, 1, 2.5, 7, 2]),
        ),
        (
            model.NumericalOperation.SUB,
            pd.Series([2, 3, -1, 2.3, -3, 0]),
        ),
        (
            model.NumericalOperation.MUL,
            pd.Series([3, -2, 0, 0.24, 10, 1]),
        ),
        (
            model.NumericalOperation.TRUEDIV,
            pd.Series([3, -2, 0, 24, 0.4, 1], dtype='float64'),
        ),
        (
            model.NumericalOperation.FLOORDIV,
            pd.Series([3, -2, 0, 23, 0, 1], dtype='float64'),
        ),
        (
            model.RelationalOperation.EQ,
            pd.Series([False, False, False, False, False, True]),
        ),
        (
            model.RelationalOperation.NE,
            pd.Series([True, True, True, True, True, False]),
        ),
        (
            model.RelationalOperation.GT,
            pd.Series([True, True, False, True, False, False]),
        ),
        (
            model.RelationalOperation.GE,
            pd.Series([True, True, False, True, False, True]),
        ),
        (
            model.RelationalOperation.LT,
            pd.Series([False, False, True, False, True, False]),
        ),
        (
            model.RelationalOperation.LE,
            pd.Series([False, False, True, False, True, True]),
        ),
    ],
)
def test_NumericalOperator(
    a: pd.Series,
    b: pd.Series,
    op: T.Union[model.RelationalOperation, model.NumericalOperation],
    exp: pd.Series,
):
    result = op(a, b)
    pd.testing.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    'data',
    [
        pd.DataFrame(
            {
                'a': [0, 1, 2],
                'b': [1, 2, 3],
                'c': [2, 3, 4],
                'd': [4, 5, 6],
            }
        ),
    ],
)
@pytest.mark.parametrize(
    'features,exp',
    [
        (['a', 'b', 'c'], pd.Series([3, 6, 9])),
        (['a', 'b'], pd.Series([1, 3, 5])),
    ],
)
def test_NewFeatureModel_calculate(
    features: T.List[str],
    data: pd.DataFrame,
    exp: pd.Series,
):
    feature = model.NewFeatureModel(
        name='test',
        features=features,
        op='add',  # type: ignore[arg-type]
        keep=False,
    )

    result = feature.calculate(data)

    pd.testing.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    'data',
    [
        pd.DataFrame(
            {
                'a': [0, 1, 2],
                'b': [1, 2, 3],
                'c': [2, 3, 4],
                'd': [4, 5, 6],
            }
        ),
    ],
)
@pytest.mark.parametrize(
    'features,exp',
    [
        (
            ['f', 'e'],
            "Missing columns ['e', 'f'] in input. "
            "Available columns: ['a', 'b', 'c', 'd'].",
        ),
    ],
)
def test_NewFeatureModel_calculate_raises(
    features: T.List[str],
    data: pd.DataFrame,
    exp: str,
):
    feature = model.NewFeatureModel(
        name='test',
        features=features,
        op='add',  # type: ignore[arg-type]
        keep=False,
    )

    with pytest.raises(ValueError) as exc_info:
        feature.calculate(data)

    assert exc_info.value.args[0] == exp
