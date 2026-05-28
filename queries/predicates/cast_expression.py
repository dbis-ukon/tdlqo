from queries.predicates.expression import Expression
from schemas.data_types.data_type import DataType


class CastExpression(Expression):
    def __init__(self, expression: Expression, data_type: DataType):
        super().__init__(f'CAST({expression.pattern()} AS {data_type.name()})', expression.pattern_map())
        self._expression = expression
        self._data_type = data_type
