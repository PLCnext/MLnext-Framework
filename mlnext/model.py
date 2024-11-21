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
        return _op(a, b)


class LogicalOperation(Operation):
    """Defines a logical operation between operands.

    Attributes:
        OR: logical or
        AND: logical and
        XOR: XOR operation

    .. versionadded:: 0.6.0
    """

    OR: str = 'or'
    AND: str = 'and'
    XOR: str = 'xor'


class NumericalOperation(Operation):
    """Defines a numerical operation between operands.

    Attributes:
        ADD: Addition
        SUB: Substraction
        MUL: Multiplication
        TRUEDIV: Division (/)
        FLOORDIV: Integer Division (//)

    .. versionadded:: 0.6.0
    """

    ADD: str = 'add'
    SUB: str = 'sub'
    MUL: str = 'mul'
    TRUEDIV: str = 'truediv'
    FLOORDIV: str = 'floordiv'


class RelationalOperation(Operation):
    """Defines a comparison between two operands.

    Attributes:
        EQ: Equality
        NE: Not equal
        GT: Greater than
        GE: Greater equal
        LT: Less than
        LE: Less equal

    .. versionadded:: 0.6.0
    """

    EQ = 'eq'
    NE = 'ne'
    GT = 'gt'
    GE = 'ge'
    LT = 'lt'
    LE = 'le'


class NewFeatureModel(BaseModel):
    """Defines new features for the :class:`FeatureCreator`.

    Attributes:
        name (str): Name of the new feature.
        features (list[str]): Name of the features to combine.
        op (LogicalOperation | NumericalOperation | RelationalOperation):
          Operation to apply to features.
        keep (bool): Whether to keep feature in the final output.
          Default: True.

    .. versionadded:: 0.6.0
    """

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
            ...,  # type: ignore[call-overload]
            min_items=2,  # type: ignore[call-arg]
            description='List of features names.',
        )
    op: T.Union[LogicalOperation, NumericalOperation, RelationalOperation] = (
        Field(
            description='Operation to apply to the features.',
        )
    )
    keep: bool = Field(
        True,
        description='Whether to keep the feature. If False, the result is '
        'only available as an intermediary feature.',
    )

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculates the new feature from the given features.

        Args:
            data (pd.DataFrame): Given features.

        Raises:
            ValueError: Raised if less than 2 features are given.

        Returns:
            pd.Series: Returns the new feature.
        """
        missing = sorted(list(set(self.features) - set(data.columns)))
        if len(missing) > 0:
            raise ValueError(
                f'Missing columns {missing} in input. '
                f'Available columns: {sorted(list(data.columns))}.'
            )

        result = data.loc[:, self.features[0]]
        for _, series in data.loc[:, self.features[1:]].items():
            result = self.op(result, series)

        return result
