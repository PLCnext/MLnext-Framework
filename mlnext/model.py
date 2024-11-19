import operator
import typing as T
from enum import Enum
from typing import TypeVar

import pandas as pd
from pydantic import __version__ as pydatic_version
from pydantic import BaseModel
from pydantic import Field

PYDANTIC_V2 = pydatic_version.startswith('2')
OperationT = TypeVar('OperationT')

__all__ = [
    'LogicalOperation',
    'NumericalOperation',
    'RelationalOperation',
    'NewFeatureModel',
]


class Operation(str, Enum):
    def __call__(self, a: OperationT, b: OperationT) -> OperationT:
        """Performs an operation on `a` and `b`.

        Args:
            a (OperationT): A.
            b (OperationT): B.

        Returns:
            OperationT: Returns the result of the operation.
        """
        _op = getattr(operator, f'__{self.value}__')
        print(a, b)
        return _op(a, b)


class LogicalOperation(Operation):
    """Defines a logical operation between operands."""

    OR: str = 'or'
    AND: str = 'and'
    XOR: str = 'xor'


class NumericalOperation(Operation):
    """Defines a numerical operation between operands."""

    ADD: str = 'add'
    SUB: str = 'sub'
    MUL: str = 'mul'
    TRUEDIV: str = 'truediv'
    FLOORDIV: str = 'floordiv'


class RelationalOperation(Operation):
    """Defines a comparison between two operands."""

    EQ = 'eq'
    NE = 'ne'
    GT = 'gt'
    GE = 'ge'
    LT = 'lt'
    LE = 'le'


class NewFeatureModel(BaseModel):
    name: str = Field(
        description='Name of the new feature.',
    )
    if PYDANTIC_V2:
        features: T.List[str] = Field(
            min_length=2,
            description='List of features names.',
        )
    else:
        features: T.List[str] = Field(  # type: ignore[no-redef]
            min_items=2,  # type: ignore[call-arg]
            description='List of features names.',
        )
    op: LogicalOperation | NumericalOperation | RelationalOperation = Field(
        description='Operation to apply to the features.',
    )
    keep: bool = Field(
        True,
        description='Whether to keep the feature. If False, the result is '
        'only available as an intermediary feature.',
    )

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        """Calculates the new feature from the given features.

        Args:
            features (pd.DataFrame): Given features.

        Raises:
            ValueError: Raised if less than 2 features are given.

        Returns:
            pd.Series: Returns the new feature.
        """

        if len(features.columns) < 2:
            raise ValueError('Expected at least 2 features.')

        result = features.iloc[:, 0]
        for _, series in features.iloc[:, 1:].items():
            result = self.op(result, series)

        return result
